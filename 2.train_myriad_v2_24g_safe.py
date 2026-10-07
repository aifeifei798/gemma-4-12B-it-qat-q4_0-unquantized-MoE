# ==============================================================================
# Myriad-MoE v2.3 [领域监督 + 24G真安全版]:
# 数据源见 DATA_TAG: v3 = myriad_tokenized_cache_v3.pt (28,000条, 专属源+中文特区+真实拒答)
# 拓扑形态: 1 共享主宰(Rank-32) + 8 大天王(Rank-16, Top-2) + 16 宗门 x 16 微专家(满血 Rank-16, Top-2 x Top-2)
# 相对 v2.2 的修复:
#   [R1] 路由监督: core_id->macro(8分类) + cluster_id->cluster(16分类) CE, 否则领域对齐只是名义
#   [R2] 显存: 4bit基座(可选) + gradient-checkpoint + BATCH=1/ACCUM=16, 基座23G bf16不可能<=22G
#   [R3] 去双重缩放: gamma=1.0, 只保留alpha/r, 原1.6%增量太弱
#   [R4] micro aux改Switch式density*p (与macro同量级), 原p_mean^2无density且0.1加权信号弱10x
#   [R5] micro改view+broadcast免repeat, 存取权重全CPU化+device自适应, DataLoader确定性seed
# v2.4 训练优化:
#   [R7] warmup+cosine调度 (替代恒定lr) + EPOCHS预算 + 240条验证集快照 + 断点含scheduler
# ==============================================================================

import os
# 【底层安全阀】根除显存碎片化 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None
    _BNB_AVAILABLE = False


# ==========================================================
# 1. 真实 Top-K 稀疏门控核 (带 Switch 负载均衡防塌陷)
# ==========================================================
class TrueTopKGating(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x_flat):
        N, _ = x_flat.shape
        logits = self.router(x_flat)  # [N, num_experts]
        # [R1] 存logits供领域监督用 (不改变返回签名, 避免破坏调用方)
        self.last_logits = logits
        probs = torch.softmax(logits.float(), dim=-1).to(logits.dtype)

        # 真正硬截断 Top-K
        topk_vals, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # 严格稀疏掩码：未选中的专家权重绝对为 0.0
        sparse_weights = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_weights)

        # Switch-Transformer 负载均衡辅助损失 (float32下计算保数值稳定)
        density = torch.zeros(self.num_experts, device=x_flat.device, dtype=torch.float32)
        density.scatter_add_(0, topk_indices[:, 0], torch.ones(N, device=x_flat.device, dtype=torch.float32))
        density = density / N
        p_mean = probs.float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * p_mean)

        return sparse_weights, topk_indices, aux_loss


# ==========================================================
# 2. 金字塔层 (大核动态路由 + 宗门条件解耦满血微专家)
# ==========================================================
class MyriadTrueRoutingLayer(nn.Module):
    def __init__(
        self,
        original_mlp,
        hidden_dim=3840,
        shared_rank=32,
        num_macro_cores=8,
        macro_rank=16,
        num_clusters=16,
        experts_per_cluster=16,
        micro_rank=16,
        device="cuda:0",
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_macro_cores = num_macro_cores
        self.macro_rank = macro_rank
        self.total_macro_rank = num_macro_cores * macro_rank  # 8 * 16 = 128
        
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.micro_rank = micro_rank
        # 总秩: 16 * 16 * 16 = 4096
        self.total_micro_rank = num_clusters * experts_per_cluster * micro_rank

        # [R3] 去双重缩放: 只保留LoRA alpha/r, gamma=1.0
        # 原 gamma=1/sqrt(3840)≈0.016 会把增量压到1.6%, 中后期欠拟合; B=0已保证零扰动, 无需再压
        self.gamma = 1.0
        self.scale_shared = 16.0 / shared_rank
        self.scale_macro = 16.0 / macro_rank
        self.scale_micro = 16.0 / micro_rank

        self.base_mlp = original_mlp

        # ----------------------------------------------------
        # L0.5: 共享主宰 (全局保底底座)
        # ----------------------------------------------------
        self.shared_lora_A = nn.Linear(hidden_dim, shared_rank, bias=False, device=device, dtype=dtype)
        self.shared_lora_B = nn.Linear(shared_rank, hidden_dim, bias=False, device=device, dtype=dtype)

        # ----------------------------------------------------
        # L1: 八大天王 (8 选 2 动态宏观路由)
        # ----------------------------------------------------
        self.router_macro = TrueTopKGating(hidden_dim, num_macro_cores, top_k=2).to(device=device, dtype=dtype)
        self.macro_lora_A = nn.Parameter(torch.empty(hidden_dim, self.total_macro_rank, device=device, dtype=dtype))
        self.macro_lora_B = nn.Parameter(torch.empty(self.total_macro_rank, hidden_dim, device=device, dtype=dtype))

        # ----------------------------------------------------
        # L2: 宗门与满血微专家 (16 宗门选 2 x 宗门内独立选 2)
        # ----------------------------------------------------
        self.router_cluster = TrueTopKGating(hidden_dim, num_clusters, top_k=2).to(device=device, dtype=dtype)
        self.router_micro = nn.Linear(hidden_dim, num_clusters * experts_per_cluster, bias=False, device=device, dtype=dtype)
        
        self.micro_lora_A = nn.Parameter(torch.empty(hidden_dim, self.total_micro_rank, device=device, dtype=dtype))
        self.micro_lora_B = nn.Parameter(torch.empty(self.total_micro_rank, hidden_dim, device=device, dtype=dtype))

        self.current_aux_loss = None
        # [R1] 供主循环做领域监督的logits缓存 ([N,8] / [N,16])
        self.current_macro_logits = None
        self.current_cluster_logits = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.shared_lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.macro_lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.micro_lora_A, a=math.sqrt(5))

        # B 矩阵置零，保障初始状态对基座模型绝对零扰动
        nn.init.zeros_(self.shared_lora_B.weight)
        nn.init.zeros_(self.macro_lora_B)
        nn.init.zeros_(self.micro_lora_B)

    def forward(self, x):
        base_out = self.base_mlp(x)
        
        B, S, D = x.shape
        N = B * S
        x_flat = x.view(N, D)

        # 1. 共享主宰
        shared_out = self.scale_shared * self.shared_lora_B(self.shared_lora_A(x_flat))

        # 2. 八大天王宏观路由 (8 选 2)
        w_macro, _, aux_macro = self.router_macro(x_flat)  # [N, 8]
        h_macro = torch.matmul(x_flat, self.macro_lora_A)  # [N, 128]
        h_macro_weighted = (h_macro.view(N, self.num_macro_cores, self.macro_rank) * 
                            w_macro.unsqueeze(-1)).view(N, self.total_macro_rank)
        macro_out = self.scale_macro * torch.matmul(h_macro_weighted, self.macro_lora_B)

        # 3. 满血微专家条件独立动态路由
        w_cluster, _, aux_cluster = self.router_cluster(x_flat)  # [N, 16]
        # [R1] 缓存监督用logits (gating内部已存last_logits, 这里直接取, 避免重复matmul)
        self.current_macro_logits = self.router_macro.last_logits
        self.current_cluster_logits = self.router_cluster.last_logits

        micro_logits = self.router_micro(x_flat).view(N, self.num_clusters, self.experts_per_cluster)  # [N, 16, 16]
        micro_probs = torch.softmax(micro_logits.float(), dim=-1).to(micro_logits.dtype)

        topk_micro_vals, topk_micro_idx = torch.topk(micro_probs, k=2, dim=-1)  # [N, 16, 2]
        topk_micro_w = topk_micro_vals / (topk_micro_vals.sum(dim=-1, keepdim=True) + 1e-8)

        local_sparse_w = torch.zeros_like(micro_probs).scatter_(-1, topk_micro_idx, topk_micro_w)
        w_joint = w_cluster.unsqueeze(-1) * local_sparse_w  # [N, 16, 16]

        h_micro = torch.matmul(x_flat, self.micro_lora_A)  # [N, 4096]
        # [R5] view+broadcast免repeat: 省一半临时显存, 数学等价 ((w*h)@B == w*(h@B))
        h_micro_sparse = (
            h_micro.view(N, self.num_clusters, self.experts_per_cluster, self.micro_rank)
            * w_joint.unsqueeze(-1)
        ).view(N, self.total_micro_rank)
        micro_out = self.scale_micro * torch.matmul(h_micro_sparse, self.micro_lora_B)

        # [R4] micro aux: 每个宗门独立Switch损失再平均, 与macro/cluster同量级(均匀~1.0)
        # 宗门c内: density_c(e)=选中次数/(N*2), p_c(e)=mean probs, aux_c=E*Σd*p; aux_micro=mean_c aux_c
        # (注: 不能直接展平到256专家做一次Switch, 否则top2+多分布求和会把均匀值抬到8)
        with torch.no_grad():
            one_hot = torch.zeros(N, self.num_clusters, self.experts_per_cluster, device=x_flat.device, dtype=torch.float32)
            one_hot.scatter_(-1, topk_micro_idx, torch.ones(N, self.num_clusters, 2, device=x_flat.device, dtype=torch.float32))
            density_micro = one_hot.sum(dim=0) / (N * 2)  # [C, E], 每行和=1
        p_mean_micro = micro_probs.float().mean(dim=0)  # [C, E], 每行和=1
        aux_micro = (torch.sum(density_micro * p_mean_micro, dim=-1) * self.experts_per_cluster).mean()

        self.current_aux_loss = aux_macro + aux_cluster + aux_micro

        # 4. LoRA增量融合输出 (gamma=1.0, 只用alpha/r缩放)
        fused_delta = shared_out + macro_out + micro_out
        return base_out + self.gamma * fused_delta.view(B, S, D)


# ==========================================================
# 3. 二进制预编译张量数据集加载器与动态 Collate
# ==========================================================
class PretokenizedMyriadDataset(Dataset):
    def __init__(self, pt_path):
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"[!] 找不到数据文件: {pt_path}，请先运行数据装配器！")
        print(f"[*] 正在载入预编译二进制语料张量: {pt_path}...")
        # 本地可信缓存, torch>=2.6默认weights_only=True会拦list/dict结构, 显式关掉
        self.records = torch.load(pt_path, map_location="cpu", weights_only=False)
        print(f"[✔] 成功载入 {len(self.records)} 条精炼多领域样本！")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def dynamic_collate_fn(batch, pad_token_id=0):
    input_ids_list = [item["input_ids"].long() for item in batch]
    labels_list = [item["labels"].long() for item in batch]

    # 动态补齐到当前批次最大长度，避免全局固定 Padding 浪费算力与显存
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )

    core_ids = torch.tensor([item["core_id"] for item in batch], dtype=torch.long)
    cluster_ids = torch.tensor([item["cluster_id"] for item in batch], dtype=torch.long)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "core_ids": core_ids,
        "cluster_ids": cluster_ids
    }


# ==========================================================
# 4. 权重持久化辅助函数 (全CPU化 + device自适应, [R5])
# ==========================================================
def _cpu_state(module):
    return {k: v.data.cpu() for k, v in module.state_dict().items()}


def extract_weights_dict(trainable_wrappers, start_layer):
    save_dict = {}
    for i, w in enumerate(trainable_wrappers):
        idx = start_layer + i
        save_dict[f"layer_{idx}_shared_lora_A"] = _cpu_state(w.shared_lora_A)
        save_dict[f"layer_{idx}_shared_lora_B"] = _cpu_state(w.shared_lora_B)
        save_dict[f"layer_{idx}_macro_lora_A"] = w.macro_lora_A.data.cpu()
        save_dict[f"layer_{idx}_macro_lora_B"] = w.macro_lora_B.data.cpu()
        save_dict[f"layer_{idx}_router_macro"] = _cpu_state(w.router_macro.router)
        save_dict[f"layer_{idx}_router_cluster"] = _cpu_state(w.router_cluster.router)
        save_dict[f"layer_{idx}_router_micro"] = _cpu_state(w.router_micro)
        save_dict[f"layer_{idx}_micro_lora_A"] = w.micro_lora_A.data.cpu()
        save_dict[f"layer_{idx}_micro_lora_B"] = w.micro_lora_B.data.cpu()
    return save_dict


def _infer_wrapper_device_dtype(wrapper, fallback_dtype):
    try:
        p = next(wrapper.parameters())
        return p.device, p.dtype
    except StopIteration:
        return torch.device("cpu"), fallback_dtype


def load_weights_dict(trainable_wrappers, start_layer, state_dict, dtype=None, device=None):
    for i, w in enumerate(trainable_wrappers):
        idx = start_layer + i
        dev, dt = _infer_wrapper_device_dtype(w, dtype or torch.bfloat16)
        if device is not None:
            dev = torch.device(device)
        if dtype is not None:
            dt = dtype
        w.shared_lora_A.load_state_dict({k: v.to(dev, dtype=dt) for k, v in state_dict[f"layer_{idx}_shared_lora_A"].items()})
        w.shared_lora_B.load_state_dict({k: v.to(dev, dtype=dt) for k, v in state_dict[f"layer_{idx}_shared_lora_B"].items()})
        w.macro_lora_A.data.copy_(state_dict[f"layer_{idx}_macro_lora_A"].to(dev, dtype=dt))
        w.macro_lora_B.data.copy_(state_dict[f"layer_{idx}_macro_lora_B"].to(dev, dtype=dt))
        w.router_macro.router.load_state_dict({k: v.to(dev, dtype=dt) for k, v in state_dict[f"layer_{idx}_router_macro"].items()})
        w.router_cluster.router.load_state_dict({k: v.to(dev, dtype=dt) for k, v in state_dict[f"layer_{idx}_router_cluster"].items()})
        w.router_micro.load_state_dict({k: v.to(dev, dtype=dt) for k, v in state_dict[f"layer_{idx}_router_micro"].items()})
        w.micro_lora_A.data.copy_(state_dict[f"layer_{idx}_micro_lora_A"].to(dev, dtype=dt))
        w.micro_lora_B.data.copy_(state_dict[f"layer_{idx}_micro_lora_B"].to(dev, dtype=dt))


# ==========================================================
# 4b. 分块前向/反向 (R6: 防显示看门狗Xid 8) + 异常抢救存盘
# 背景: 训练卡同时带桌面时, S=448长序列的整批forward/backward burst
#       可饿死显示通道触发RC watchdog (Xid 8, launch timed out)。
#       按MAX_TOKENS切块 + 块间1-token交叠, LM/Sup与整批数学等价, Aux取块平均近似。
# ==========================================================
_LIVE = {}  # 主循环注册可抢救状态, __main__异常时用


def _emergency_save():
    if "wrappers" not in _LIVE:
        print("[!] 主循环未启动, 无可抢救状态")
        return
    try:
        weights_dict = extract_weights_dict(_LIVE["wrappers"], _LIVE["start_layer"])
        torch.save({
            "step": _LIVE.get("step", 0),
            "batch_offset": _LIVE.get("batch", 0),
            "optimizer": _LIVE["optimizer"].state_dict(),
            "scheduler": _LIVE["scheduler"].state_dict() if "scheduler" in _LIVE else None,
            "weights": weights_dict,
            "version": "v3.0",
            "emergency": True,
        }, _LIVE["checkpoint_file"])
        torch.save(weights_dict, _LIVE["infer_export_path"])
        print(f"[✔] 抢救成功 step={_LIVE.get('step', 0)}")
    except Exception as e:
        print(f"[!] 抢救失败 (CUDA上下文可能已损坏): {type(e).__name__}: {e}")


def forward_backward_batch(model, trainable_wrappers, num_layers, input_ids, labels,
                           core_ids, cluster_ids, pad_token_id,
                           aux_weight, sup_weight, backward_scale, max_tokens=256,
                           sup_response_only=True):
    """单batch分块前向+逐块反向, 梯度已按backward_scale累加.
    返回 (lm, aux, sup, acc_macro, acc_clust, S, n_chunks) 供日志 (均为标量tensor)."""
    B, S = input_ids.shape
    device = input_ids.device
    # max_tokens<=0: 不分块 (整批, 与v2.3行为一致, 保真但长burst可能再触发看门狗)
    starts = [0] if (not max_tokens or max_tokens <= 0) else list(range(0, S, max_tokens))
    # --- 第一遍: 只数token (免模型, 供跨块加权) ---
    slices, lm_n_total, sup_n_total = [], 0, 0
    for st in starts:
        en = min(st + max_tokens, S)
        ctx = max(st - 1, 0)  # 交叠1 token做左上下文, 不计分
        slices.append((ctx, st, en))
        lab_c = labels[:, st:en]
        if ctx == st:
            lm_n_total += int((lab_c[:, 1:] != -100).sum())
        else:
            lm_n_total += int((lab_c != -100).sum())
        resp = (labels[:, st:en] != -100) if sup_response_only else (input_ids[:, st:en] != pad_token_id)
        sup_n_total += int(resp.sum())
    lm_n_total = max(lm_n_total, 1)
    sup_n_total = max(sup_n_total, 1)
    n_chunks = len(slices)

    lm_acc, aux_acc = 0.0, 0.0
    sup_m_acc, sup_c_acc = 0.0, 0.0
    acc_m_ok, acc_c_ok = 0.0, 0.0
    for ctx, st, en in slices:
        inp_c = input_ids[:, ctx:en]
        lab_c = labels[:, st:en]
        outputs = model(input_ids=inp_c, use_cache=False)
        logits_c = outputs.logits
        if ctx == st:  # 首块: 内部shift, 尾label留给下一块的交叠打分
            sl_c, lb_c = logits_c[:, :-1, :].contiguous(), lab_c[:, 1:].contiguous()
        else:          # 交叠块: 交叠token只做上下文, lab全量恰好对齐
            sl_c, lb_c = logits_c[:, :-1, :].contiguous(), lab_c.contiguous()
        lm_c = F.cross_entropy(sl_c.view(-1, sl_c.size(-1)), lb_c.view(-1),
                               ignore_index=-100, reduction="sum")

        aux_c = sum(w.current_aux_loss for w in trainable_wrappers) / num_layers

        # 路由logits行号: batch-major下sup token(st..en-1)对应行
        Sc = en - ctx
        rows = (torch.arange(B, device=device)[:, None] * Sc
                + torch.arange(st - ctx, Sc, device=device)[None, :]).reshape(-1)
        sup_tok = input_ids[:, st:en].reshape(-1)
        sup_lab = labels[:, st:en].reshape(-1)
        sup_mask = (sup_tok != pad_token_id)
        if sup_response_only:
            sup_mask = sup_mask & (sup_lab != -100)
        core_c = core_ids[:, None].expand(B, en - st).reshape(-1)
        clus_c = cluster_ids[:, None].expand(B, en - st).reshape(-1)
        sup_m_c = sup_c_c = 0.0
        if int(sup_mask.sum().item()) > 0:
            for w in trainable_wrappers:
                ml = w.current_macro_logits[rows]
                cl = w.current_cluster_logits[rows]
                sup_m_c = sup_m_c + F.cross_entropy(ml[sup_mask].float(), core_c[sup_mask], reduction="sum")
                sup_c_c = sup_c_c + F.cross_entropy(cl[sup_mask].float(), clus_c[sup_mask], reduction="sum")
                acc_m_ok = acc_m_ok + (ml[sup_mask].argmax(-1) == core_c[sup_mask]).float().sum()
                acc_c_ok = acc_c_ok + (cl[sup_mask].argmax(-1) == clus_c[sup_mask]).float().sum()

        ((lm_c / lm_n_total
          + aux_weight * (aux_c / n_chunks)
          + sup_weight * ((sup_m_c + sup_c_c) / (2 * num_layers * sup_n_total))
         * backward_scale)).backward()

        lm_acc = lm_acc + lm_c.detach()
        aux_acc = aux_acc + aux_c.detach()
        sup_m_acc = sup_m_acc + (sup_m_c.detach() if torch.is_tensor(sup_m_c) else 0.0)
        sup_c_acc = sup_c_acc + (sup_c_c.detach() if torch.is_tensor(sup_c_c) else 0.0)

    lm_loss = lm_acc / lm_n_total
    aux_loss = aux_acc / n_chunks
    sup_loss = (sup_m_acc + sup_c_acc) / (2 * num_layers * sup_n_total)
    acc_macro = acc_m_ok / (num_layers * sup_n_total)
    acc_clust = acc_c_ok / (num_layers * sup_n_total)
    # [R7b] 整批无response token时(如prompt占满448)累加器是Python float, 统一转tensor防.item()崩
    to_t = lambda v: v if torch.is_tensor(v) else torch.tensor(v, device=device)
    return to_t(lm_loss), to_t(aux_loss), to_t(sup_loss), to_t(acc_macro), to_t(acc_clust), S, n_chunks


# ==========================================================
# 5. 主训练流水线
# ==========================================================
@torch.no_grad()
def eval_snapshot(model, trainable_wrappers, num_layers, valloader, device, pad_token_id, max_batches=60):
    """[R7] 验证集快照: LM(response)/sup-acc(response)/aux均值, 不开梯度, 不改训练状态."""
    was_training = [w.training for w in trainable_wrappers]
    model.eval()
    lm_s, aux_s, sup_m_s, sup_c_s = 0.0, 0.0, 0.0, 0.0
    am_ok, ac_ok, n_tok, n_b = 0.0, 0.0, 0, 0
    for batch in valloader:
        if n_b >= max_batches:
            break
        n_b += 1
        ids = batch["input_ids"].to(device)
        lab = batch["labels"].to(device)
        core = batch["core_ids"].to(device)
        clu = batch["cluster_ids"].to(device)
        out = model(input_ids=ids, use_cache=False).logits
        lm_s += F.cross_entropy(out[..., :-1, :].contiguous().view(-1, out.size(-1)),
                                lab[..., 1:].contiguous().view(-1),
                                ignore_index=-100).item()
        aux_s += float(sum(w.current_aux_loss for w in trainable_wrappers).item() / num_layers)
        B, S = ids.shape
        mk = ((lab != -100) & (ids != pad_token_id)).view(-1)
        if int(mk.sum().item()) > 0:
            ce_ = core[:, None].expand(B, S).reshape(-1)
            cu_ = clu[:, None].expand(B, S).reshape(-1)
            for w in trainable_wrappers:
                ml = w.current_macro_logits
                cl = w.current_cluster_logits
                sup_m_s += F.cross_entropy(ml[mk].float(), ce_[mk]).item()
                sup_c_s += F.cross_entropy(cl[mk].float(), cu_[mk]).item()
                am_ok += (ml[mk].argmax(-1) == ce_[mk]).float().sum().item()
                ac_ok += (cl[mk].argmax(-1) == cu_[mk]).float().sum().item()
            n_tok += int(mk.sum().item())
    n_b = max(n_b, 1)
    n_tok = max(n_tok, 1)
    for w, t in zip(trainable_wrappers, was_training):
        w.train(t)
    return (lm_s / n_b, aux_s / n_b,
            (sup_m_s + sup_c_s) / (2 * num_layers * n_b),
            am_ok / (num_layers * n_tok), ac_ok / (num_layers * n_tok))


def main():
    # ---- 超参 (v3: 数据见1.prepare_myriad_v3.py, 28k/8x3500/16x1750, 中文特区+真实拒答) ----
    SEED = 42
    DATA_TAG = "v3"           # 数据/断点/权重文件名后缀 (v2用v2, 互不干扰)
    model_id = "../gemma-4-12B-it-qat-q4_0-unquantized"
    data_cache_path = f"myriad_tokenized_cache_{DATA_TAG}.pt"
    checkpoint_file = f"checkpoint_{DATA_TAG}.pt"
    infer_export_path = f"myriad_moe_hierarchical_weights_{DATA_TAG}.pt"
    BATCH_SIZE = 1          # [R2] 2->1: 基座23G bf16时B=2必爆, B=1+ACCUM=16保持等效16
    ACCUM_STEPS = 16
    AUX_WEIGHT = 0.005
    SUP_WEIGHT = 0.3         # [R1b] 0.1->0.3: LM降到0.27后监督信号被淹没, 领域命中上不去
    SUP_RESPONSE_ONLY = True # [R1b] 只拿response token监督路由; prompt模板各领域完全一样, 会给矛盾标签把路由拉散
    QUANT_MODE = "4bit"     # [R2] "4bit"(已实测更快更省: 前向89ms/基座7.18G)/"8bit"(175ms/12.09G)/"bf16"(需32G卡)
    USE_8BIT_OPTIM = True   # [R2] PagedAdamW8bit, 392M参数省约4G; 不可用自动回退
    USE_GRAD_CKPT = True    # [R2] 省activation数GB
    MAX_OPT_STEPS = 0       # 0=按EPOCHS跑; 预飞验证时可设60
    EPOCHS = 3            # [R7] 训练预算(轮); 单轮~1485步, 3轮约4455步(含已跑1500)
    WARMUP_STEPS = 100    # [R7] 线性warmup步数 (路由随机初值, 恒定lr是当前最大短板)
    LR_MIN = 3e-5         # [R7] cosine下限
    VAL_SIZE = 240        # [R7] 验证集条数(~1%); 0=关闭验证
    VAL_EVERY = 100       # [R7] 每N个优化器步验证一次
    MAX_TOKENS_PER_FORWARD = 256  # [R6] 单次forward上限防看门狗(Xid 8); 设0=不分块(保真但长burst有风险)
    # 注: 分块后块2+的token丢左文attention上下文(18.8%样本尾部受影响, LM 4.59vs整批2.67@S=448);
    #     路由等逐token算子仍精确。稳定优先默认256, 若复现崩溃且日志S栏多为x1短批, 则与分块无关, 需查驱动/内核。

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("=" * 85)
    print("【Myriad-MoE v3.0】启动")
    print(f"   装载语料: {data_cache_path}")
    print("   拓扑架构: 8 大天王 (Rank-16) + 16 宗门 x 16 微专家 (满血 Rank-16)")
    print(f"   批量: B={BATCH_SIZE} x ACCUM={ACCUM_STEPS} = 等效{BATCH_SIZE*ACCUM_STEPS} | aux_w={AUX_WEIGHT} sup_w={SUP_WEIGHT}")
    print(f"   量化: {QUANT_MODE} (bnb可用: {_BNB_AVAILABLE}) | 8bit优化器: {USE_8BIT_OPTIM} | grad_ckpt: {USE_GRAD_CKPT} | seed: {SEED}")
    print("   注意: bf16基座单文件23G, 必须量化才进得去24G")
    print("=" * 85)

    dtype = torch.bfloat16
    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print("[*] 正在加载 Gemma-4-12B 语言模型...")
    if QUANT_MODE in ("4bit", "8bit") and _BNB_AVAILABLE:
        if QUANT_MODE == "8bit":
            # [R2] 已实测: 5090D上基座12.09G, 2层+2step峰值18.8G, 外推12层约21G, 进得去24G
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        else:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map=device,
            attn_implementation="sdpa",
        )
        print(f"[✔] {QUANT_MODE}量化基座加载成功 (训练增量仍bf16)")
    else:
        if QUANT_MODE in ("4bit", "8bit") and not _BNB_AVAILABLE:
            print("[!] 未装bitsandbytes, 回退bf16全量 (需~23G, 24G卡会爆, 建议pip install bitsandbytes)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            attn_implementation="sdpa"
        )

    # 释放多模态显存冗余
    if hasattr(model.model, "embed_vision"):
        del model.model.embed_vision
    if hasattr(model.model, "embed_audio"):
        del model.model.embed_audio
    gc.collect()
    torch.cuda.empty_cache()

    for p in model.parameters():
        p.requires_grad = False

    text_model = model.model.language_model
    all_layers = text_model.layers
    text_config = getattr(model.config, "text_config", model.config)
    hidden_dim = getattr(text_config, "hidden_size", 3840)

    START_LAYER = 18
    NUM_MOE_LAYERS = 12
    END_LAYER = START_LAYER + NUM_MOE_LAYERS

    trainable_wrappers = []
    print(f"[*] 正在为 Layer {START_LAYER} ~ {END_LAYER-1} 注入金字塔路由层...")

    for idx in range(START_LAYER, END_LAYER):
        layer = all_layers[idx]
        wrapper = MyriadTrueRoutingLayer(
            original_mlp=layer.mlp,
            hidden_dim=hidden_dim,
            shared_rank=32,
            num_macro_cores=8,
            macro_rank=16,
            num_clusters=16,
            experts_per_cluster=16,
            micro_rank=16,
            device=device,
            dtype=dtype
        )
        layer.mlp = wrapper
        trainable_wrappers.append(wrapper)

    trainable_params = [p for w in trainable_wrappers for p in w.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[*] 可训练参数: {n_trainable/1e6:.1f}M")

    # [R2] gradient checkpointing省activation
    if USE_GRAD_CKPT:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        print("[✔] gradient checkpointing已开启")

    _use_8bit_optim = False
    if USE_8BIT_OPTIM:
        try:
            import bitsandbytes as _bnb
            optimizer = _bnb.optim.PagedAdamW8bit(trainable_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
            _use_8bit_optim = True
            print("[✔] PagedAdamW8bit已启用")
        except Exception as e:
            print(f"[!] 8bit优化器不可用 ({type(e).__name__}), 回退fp32 AdamW")
    if not _use_8bit_optim:
        try:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                fused=torch.cuda.is_available()
            )
        except TypeError:
            optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

    # 断点恢复
    start_step = 0
    start_batch_offset = 0
    _sched_state = None

    if os.path.exists(checkpoint_file):
        print(f"\n[恢复] 发现断点: {checkpoint_file}")
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        start_step = ckpt.get("step", 0)
        start_batch_offset = ckpt.get("batch_offset", 0)
        _sched_state = ckpt.get("scheduler")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[!] 优化器状态不兼容 (如fp32<->8bit切换), 从零开始优化器: {type(e).__name__}")
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        load_weights_dict(trainable_wrappers, START_LAYER, ckpt["weights"])
        print(f"[✔] 续接 Step {start_step} (跳过前 {start_batch_offset} 批次, 同seed下顺序确定)\n")
    else:
        print("\n[*] 全新起跑，未检测到断点。\n")

    _LIVE.update({"wrappers": trainable_wrappers, "start_layer": START_LAYER,
                  "optimizer": optimizer, "step": start_step, "batch": start_batch_offset,
                  "checkpoint_file": checkpoint_file, "infer_export_path": infer_export_path})
    # [R5] 确定性shuffle: 同seed重启后顺序一致, batch_offset续跑才有意义
    # [R7] 验证集: 确定性随机抽VAL_SIZE条, 训练集不再见它们
    g = torch.Generator()
    g.manual_seed(SEED)
    dataset = PretokenizedMyriadDataset(data_cache_path)
    if VAL_SIZE > 0:
        gv = torch.Generator()
        gv.manual_seed(SEED + 1)
        perm = torch.randperm(len(dataset), generator=gv).tolist()
        train_set, val_set = Subset(dataset, perm[VAL_SIZE:]), Subset(dataset, perm[:VAL_SIZE])
    else:
        train_set, val_set = dataset, None
    dataloader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_token_id=pad_token_id)
    )
    valloader = (DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0,
                            collate_fn=lambda b: dynamic_collate_fn(b, pad_token_id=pad_token_id))
                 if val_set is not None else None)

    # [R7] warmup+cosine调度: 预算=EPOCHS轮; 旧断点无scheduler状态则按step快进对齐
    steps_per_epoch = max(len(dataloader) // ACCUM_STEPS, 1)
    TOTAL_STEPS = (EPOCHS * steps_per_epoch) if EPOCHS else 0
    # [R7b] warmup+cosine单函数调度 (替代SequentialLR: 后者在断点load后step()不推进lr, 已实锤)
    def _lr_factor(s, warmup=WARMUP_STEPS, total=TOTAL_STEPS, floor=LR_MIN / 3e-4):
        if s < warmup:
            return 0.05 + 0.95 * s / warmup
        if total <= warmup or s >= total:
            return floor
        prog = (s - warmup) / (total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_factor)
    if _sched_state is not None:
        try:
            scheduler.load_state_dict(_sched_state)
        except Exception as e:
            print(f"[!] scheduler状态不兼容, 按step快进: {e}")
            _sched_state = None
    if _sched_state is None and start_step > 0:
        for _ in range(start_step):
            scheduler.step()
    print(f"[*] scheduler就绪: last_epoch={scheduler.last_epoch}, "
          f"lr={scheduler.get_last_lr()[0]:.1e} (恢复态={'断点' if _sched_state is not None else '快进/全新'})")
    _LIVE["scheduler"] = scheduler
    print(f"[*] 训练预算: EPOCHS={EPOCHS}, 单轮约{steps_per_epoch}步, "
          f"总计{TOTAL_STEPS if TOTAL_STEPS else '无限'}步 (当前step={start_step}, lr={scheduler.get_last_lr()[0]:.1e})")

    # batch_offset恒为绝对坐标 (跨轮累加), 重启靠它跳过, 无需清零

    # (LM/CE统一走forward_backward_batch内F.cross_entropy, 权重见AUX/SUP)
    model.eval()
    for w in trainable_wrappers:
        w.train()

    step = start_step
    global_batch_counter = 0
    t0 = time.time()
    finished = False

    # [R7c] 真多轮: dataloader迭代器耗尽 = 跑完一遍, 重开进入下一轮 (同一generator续行, 顺序更新但确定);
    # global_batch_counter恒为绝对坐标, 跨轮/重启都不重置, 跳过与存盘天然正确
    while not finished:
        for batch in dataloader:
            global_batch_counter += 1
            if global_batch_counter <= start_batch_offset:
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            core_ids = batch["core_ids"].to(device, non_blocking=True)       # [B] 0~7
            cluster_ids = batch["cluster_ids"].to(device, non_blocking=True) # [B] 0~15

            # [R6] 分块前向/反向: 长序列不再一次打满, 显示通道有机会插空, 看门狗不杀
            lm_loss, aux_loss, sup_loss, acc_macro, acc_clust, cur_S, n_chunks = forward_backward_batch(
                model, trainable_wrappers, NUM_MOE_LAYERS, input_ids, labels,
                core_ids, cluster_ids, pad_token_id,
                AUX_WEIGHT, SUP_WEIGHT, 1.0 / ACCUM_STEPS,
                max_tokens=MAX_TOKENS_PER_FORWARD, sup_response_only=SUP_RESPONSE_ONLY)
            _LIVE["batch"] = global_batch_counter

            if global_batch_counter % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                _LIVE["step"] = step

                if step % 5 == 0:
                    elapsed = time.time() - t0
                    vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    vram_peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                    disp_total = TOTAL_STEPS if TOTAL_STEPS else len(dataloader) // ACCUM_STEPS
                    print(f"Step [{step:04d}/{disp_total:04d}] | S:{cur_S}x{n_chunks} | LM: {lm_loss.item():.4f} | Aux: {aux_loss.item():.4f} | Sup: {sup_loss.item():.4f} (acc_macro {acc_macro.item():.2%} clust {acc_clust.item():.2%}) | lr:{scheduler.get_last_lr()[0]:.1e} | 显存: {vram_mb:.0f}MB (峰值: {vram_peak:.0f}MB) | 耗时: {elapsed:.2f}s")
                    if torch.cuda.is_available() and vram_peak > 23.5 * 1024:
                        print("[!] 峰值超23.5G: 请确认4bit已生效, 或再降BATCH/开offload")
                    t0 = time.time()

                # [R7] 验证快照: 只看数, 不动权重
                if VAL_EVERY and valloader is not None and step % VAL_EVERY == 0:
                    vlm, vaux, vsup, vam, vac = eval_snapshot(
                        model, trainable_wrappers, NUM_MOE_LAYERS, valloader, device, pad_token_id)
                    print(f"[val {step:04d}] LM:{vlm:.4f} Aux:{vaux:.4f} Sup:{vsup:.4f} "
                          f"(acc_macro {vam:.2%} clust {vac:.2%})")

                # 每 50 步保存状态断点
                if step > 0 and step % 50 == 0:
                    print(f"\n[保存 Step {step:04d}] ...")
                    weights_dict = extract_weights_dict(trainable_wrappers, START_LAYER)
                    torch.save({
                        "step": step,
                        "batch_offset": global_batch_counter,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "weights": weights_dict,
                        "version": "v3.0",
                    }, checkpoint_file)
                    torch.save(weights_dict, infer_export_path)
                    print(f"   断点 -> {checkpoint_file} | 推理权重 -> {infer_export_path}\n")

                # 预飞截断: 到达MAX_OPT_STEPS即存盘退出 (0=按EPOCHS)
                if MAX_OPT_STEPS and step >= MAX_OPT_STEPS:
                    print(f"\n[预飞] 已达 MAX_OPT_STEPS={MAX_OPT_STEPS}, 存盘退出 (设为0按EPOCHS)")
                    weights_dict = extract_weights_dict(trainable_wrappers, START_LAYER)
                    torch.save({
                        "step": step,
                        "batch_offset": global_batch_counter,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "weights": weights_dict,
                        "version": "v3.0",
                    }, checkpoint_file)
                    torch.save(weights_dict, infer_export_path)
                    finished = True
                    break

                # [R7] 预算耗尽即收工
                if TOTAL_STEPS and step >= TOTAL_STEPS:
                    print(f"\n[收工] 已达预算 TOTAL_STEPS={TOTAL_STEPS}, 存盘退出")
                    weights_dict = extract_weights_dict(trainable_wrappers, START_LAYER)
                    torch.save({
                        "step": step,
                        "batch_offset": global_batch_counter,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "weights": weights_dict,
                        "version": "v3.0",
                    }, checkpoint_file)
                    torch.save(weights_dict, infer_export_path)
                    finished = True
                    break

        if finished:
            break
        # 整轮跑完但预算没用完 -> 开下一轮 (首轮跳过量已消费, 清零)
        start_batch_offset = 0
        print(f"\n[新epoch] 跑完一遍 ({global_batch_counter} batches), 开下一轮, step从{step}续\n")

    print("\n训练结束, 固化最终权重...")
    final_weights = extract_weights_dict(trainable_wrappers, START_LAYER)
    torch.save(final_weights, infer_export_path)
    print(f"[✔] 最终权重: {infer_export_path}")
    print("=" * 85)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 用户停止, 抢救断点...")
        _emergency_save()
        raise SystemExit(130)
    except Exception:
        print("\n[异常] 训练崩溃, 抢救断点...")
        _emergency_save()
        raise