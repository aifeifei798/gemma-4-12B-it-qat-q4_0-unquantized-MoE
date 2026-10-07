#!/usr/bin/env python3
# ==============================================================================
# 3.infer_probe.py -- Myriad-MoE v2.3 推理 + LoRA/微核探测器
#   推理:  基座(4bit默认) + myriad_moe_hierarchical_weights_v2.pt, 按训练格式续写
#   探测:  每层包揽 (a)LoRA分支是否学到东西 (b)8天王/16宗门/微专家每token路由 (c)领域对齐判定
# 用法:
#   python3 3.infer_probe.py --prompt "写个快排" --domain 0 --max-new-tokens 64
#   python3 3.infer_probe.py --prompt "..." --probe-only   # 只探测不生成
# ==============================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, gc, torch, torch.nn.functional as F
import importlib.util

TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2.train_myriad_v2_24g_safe.py")
spec = importlib.util.spec_from_file_location("trainmod", TRAIN_SCRIPT)
trainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainmod)
from transformers import AutoModelForCausalLM, AutoTokenizer

CORE_NAMES = ["代码工程Code", "严密数学Math", "自然科学Science", "文学创意Creative_Arts",
              "商务对话Business_Dialogue", "逻辑思辨Logic_Philosophy",
              "严格约束Constraint_Rules", "中文特区Chinese_Slots"]
# 宗门名 (v3: 14/15 改为中文创作/中文问答, 见1.prepare_myriad_v3.py)
CLUSTER_NAMES = ["语法算法", "系统工程", "数值代数", "文字推导",
                 "基础物化", "实验科学", "长篇叙事", "诗歌意象",
                 "专业问答", "公文摘要", "因果推演", "哲学批判",
                 "格式规范", "负向安全", "中文创作", "中文问答"]
CLUSTER_PARENT = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
# 英文名 (与 web/src/i18n.js CLAN_EN 同源, 真源见 domains.yaml)
CORE_EN = ["Code", "Math", "Science", "Creative_Arts", "Business_Dialogue",
           "Logic_Philosophy", "Constraint_Rules", "Chinese_Slots"]
CLUSTER_EN = ["Syntax & Algorithms", "Systems Engineering", "Numerical Algebra", "Word Problems",
              "Basic Physics & Chemistry", "Experimental Science", "Long-form Narrative", "Poetry & Imagery",
              "Expert QA", "Summarization", "Causal Reasoning", "Critical Thinking",
              "Format Compliance", "Safety Refusals", "Chinese Writing", "Chinese QA"]


def _load_domains():
    """domains.yaml 是唯一真源; 加载成功则覆盖上面硬编码 (字符串保持旧格式, 接口不破)."""
    global CORE_NAMES, CLUSTER_NAMES, CLUSTER_PARENT, CORE_EN, CLUSTER_EN
    import yaml
    fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domains.yaml")
    try:
        with open(fp, encoding="utf-8") as f:
            dom = yaml.safe_load(f)
        cores = sorted(dom.get("macro_cores", []), key=lambda c: c["id"])
        clus = sorted(dom.get("clusters", []), key=lambda c: c["id"])
        if len(cores) == 8 and len(clus) == 16:
            CORE_EN = [c["en"] for c in cores]
            CORE_NAMES = [f'{c["zh"]}{c["en"]}' for c in cores]
            CLUSTER_EN = [c["en"] for c in clus]
            CLUSTER_NAMES = [c["zh"] for c in clus]
            CLUSTER_PARENT = [c["parent"] for c in clus]
    except Exception as e:
        print(f"[!] domains.yaml 加载失败, 用内置名: {e}")


try:
    _load_domains()
except Exception:
    pass


def cluster_label(i: int) -> str:
    return f"宗门{i}·{CLUSTER_NAMES[i]}({CORE_NAMES[CLUSTER_PARENT[i]]})"


def load_system(model_id="../gemma-4-12B-it-qat-q4_0-unquantized",
                weight_path="myriad_moe_hierarchical_weights_v3.pt",
                quant="4bit", device="cuda:0", dtype=torch.bfloat16,
                load_processor=True):
    from transformers import BitsAndBytesConfig
    print(f"[*] 基座加载 ({quant})...")
    if quant == "8bit":
        qc = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    else:
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=qc,
                                                device_map=device, attn_implementation="sdpa")
    # 多模态塔常驻 (图/音/视频统一模型, 4bit量化后增量可控; 不再删除)
    gc.collect(); torch.cuda.empty_cache()
    for p in model.parameters():
        p.requires_grad = False
    tm = model.model.language_model
    hidden = getattr(model.config.text_config, "hidden_size", 3840)
    wrappers = []
    for idx in range(18, 30):
        w = trainmod.MyriadTrueRoutingLayer(tm.layers[idx].mlp, hidden_dim=hidden, device=device, dtype=dtype)
        tm.layers[idx].mlp = w
        wrappers.append(w)
    print(f"[*] 权重载入 {weight_path} ...")
    sd = torch.load(weight_path, map_location="cpu", weights_only=False)
    trainmod.load_weights_dict(wrappers, 18, sd)
    tok = AutoTokenizer.from_pretrained(model_id)
    proc = None
    if load_processor:
        try:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(model_id)
            print(f"[*] 多模态处理器: {type(proc).__name__} (图/音/视频)")
        except Exception as e:
            print(f"[!] 处理器加载失败 (纯文本模式): {e}")
    model.eval()
    return model, tok, wrappers, proc


@torch.no_grad()
def probe_forward(model, wrappers, input_ids, pad_id=0, **mkwargs):
    """单次prefill, 抓每层输入, 返回每层诊断. 不开梯度, 不改模型.
    mkwargs: 多模态透传 (pixel_values/input_features等, 与prompt对齐的张量)."""
    captured = []
    handles = [w.register_forward_hook(lambda mod, inp, out, _c=captured: _c.append(inp[0].detach()))
               for w in wrappers]
    try:
        model(input_ids=input_ids, use_cache=False, **mkwargs)
    finally:
        for h in handles:
            h.remove()
    return [diagnose_layer(w, x) for w, x in zip(wrappers, captured)]


@torch.no_grad()
def _steer_mask(logits, disabled, n):
    dis = [d for d in (disabled or []) if 0 <= d < n]
    if dis and len(dis) < n:
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, dis] = True
        logits = logits.masked_fill(mask, float("-inf"))
    return logits


@torch.no_grad()
def diagnose_layer(w, x):
    """复刻wrapper前向的路由部分, 返回可读诊断 (与训练前向数学一致, 含STEER操控)."""
    B, S, D = x.shape
    N = B * S
    xf = x.view(N, D)
    base_out = w.base_mlp(x).view(N, D)

    # --- STEER (与 wrapper.forward 同逻辑, 保证面板=生成) ---
    _t = max(float(getattr(w, "steer_temp", 1.0) or 1.0), 1e-3)
    _off_all = bool(getattr(w, "steer_disable_all", False))
    _off_sh = _off_all or bool(getattr(w, "steer_disable_shared", False))
    _off_ma = _off_all or bool(getattr(w, "steer_disable_macro", False))
    _off_mi = _off_all or bool(getattr(w, "steer_disable_micro", False))
    _dm = getattr(w, "steer_disabled_macros", []) or []
    _dc = getattr(w, "steer_disabled_clusters", []) or []

    # --- L0.5 shared ---
    if _off_sh:
        shared = torch.zeros(N, D, dtype=torch.float32)
    else:
        shared = (w.scale_shared * w.shared_lora_B(w.shared_lora_A(xf))).to(torch.float32)

    # --- L1 macro (8选2) ---
    ml = _steer_mask(w.router_macro.router(xf).float() / _t, _dm, 8)
    mp = torch.softmax(ml, -1)
    mv, mi = torch.topk(mp, k=2, dim=-1)
    mw = mv / (mv.sum(-1, keepdim=True) + 1e-8)
    sparse_macro = torch.zeros_like(mp).scatter_(-1, mi, mw)  # 与训练一致: 稀疏[N,8]
    if _off_ma:
        macro = torch.zeros(N, D, dtype=torch.float32)
    else:
        h = (xf.to(w.macro_lora_A.dtype) @ w.macro_lora_A).view(N, w.num_macro_cores, w.macro_rank)
        macro = (w.scale_macro * (h * sparse_macro.to(h.dtype).unsqueeze(-1)).view(N, -1) @ w.macro_lora_B).to(torch.float32)

    # --- L2 cluster (16选2) + micro (宗门内16选2) ---
    cl = _steer_mask(w.router_cluster.router(xf).float() / _t, _dc, 16)
    cp = torch.softmax(cl, -1)
    cv, ci = torch.topk(cp, k=2, dim=-1)
    cw = (cv / (cv.sum(-1, keepdim=True) + 1e-8))
    wcl = torch.zeros_like(cp).scatter_(-1, ci, cw)
    if _off_mi:
        micro = torch.zeros(N, D, dtype=torch.float32)
        ti = torch.zeros(N, w.num_clusters, 2, dtype=torch.long)
    else:
        mlog = w.router_micro(xf).view(N, w.num_clusters, w.experts_per_cluster).float() / _t
        mprob = torch.softmax(mlog, -1)
        tv, ti = torch.topk(mprob, k=2, dim=-1)
        tw = tv / (tv.sum(-1, keepdim=True) + 1e-8)
        local = torch.zeros_like(mprob).scatter_(-1, ti, tw)
        joint = wcl.unsqueeze(-1) * local
        hm = (xf.to(w.micro_lora_A.dtype) @ w.micro_lora_A).view(N, w.num_clusters, w.experts_per_cluster, w.micro_rank)
        micro = (w.scale_micro * (hm * joint.to(hm.dtype).unsqueeze(-1)).view(N, -1) @ w.micro_lora_B).to(torch.float32)

    base = base_out.to(torch.float32)
    e = lambda t: t.norm(dim=-1)  # [N]
    return {
        "macro_idx": mi.cpu(), "macro_w": mw.cpu(), "macro_pred": mi[:, 0].cpu(),
        "cluster_idx": ci.cpu(), "cluster_pred": ci[:, 0].cpu(),
        "micro_idx": ti.cpu(),  # [N,16,2] 每宗门Top2
        "n_shared": e(shared), "n_macro": e(macro), "n_micro": e(micro), "n_base": e(base),
        "b_shared": w.shared_lora_B.weight.detach().float().norm().item(),
        "b_macro": w.macro_lora_B.detach().float().norm().item(),
        "b_micro": w.micro_lora_B.detach().float().norm().item(),
    }


def report(diags, tok, ids, expect_core=None, top_tokens=8):
    print("\n" + "=" * 78 + "\n[探测报告]")
    # --- LoRA存活 ---
    print("--- (a) LoRA分支B矩阵范数 (0=该分支没学到东西) ---")
    for i, d in enumerate(diags):
        print(f"  L{18+i}: shared_B={d['b_shared']:.4f} macro_B={d['b_macro']:.4f} micro_B={d['b_micro']:.4f}")
    # --- 能量占比 ---
    import torch as _t
    ns = _t.stack([d["n_shared"] for d in diags]).mean(0)
    nm = _t.stack([d["n_macro"] for d in diags]).mean(0)
    nu = _t.stack([d["n_micro"] for d in diags]).mean(0)
    nb = _t.stack([d["n_base"] for d in diags]).mean(0)
    print(f"--- (b) 12层平均增量/基座能量比: shared {ns.mean()/nb.mean():.3%} "
          f"macro {nm.mean()/nb.mean():.3%} micro {nu.mean()/nb.mean():.3%} ---")
    # --- 路由直方图 (prompt token聚合) ---
    import collections
    mh = collections.Counter(sum([d["macro_pred"].tolist() for d in diags], []))
    ch = collections.Counter(sum([d["cluster_pred"].tolist() for d in diags], []))
    print(f"--- (c) 天王Top1分布(12层xT): "
          + " ".join(f"{CORE_NAMES[k]}:{v}" for k, v in sorted(mh.items())) + " ---")
    print(f"    宗门Top1分布: "
          + " ".join(f"{CLUSTER_NAMES[k]}:{v}" for k, v in sorted(ch.items())) + " ---")
    if expect_core is not None:
        hit = mh.get(expect_core, 0) / max(sum(mh.values()), 1)
        print(f"    期望领域 {expect_core}={CORE_NAMES[expect_core]}, Top1命中率 {hit:.1%} "
              f"{'对齐' if hit > 0.3 else '未对齐/待训'}")
    # --- 逐token路由 (前top_tokens个) ---
    print(f"--- (d) 前{top_tokens}个token的路由 (天王idx:权重 | 宗门Top2) ---")
    toks = [tok.decode([t]) for t in ids[0, :top_tokens].tolist()]
    d0 = diags[len(diags) // 2]  # 看中间层代表
    for i, t in enumerate(toks):
        mi = d0["macro_idx"][i].tolist()
        mw = [f"{v:.2f}" for v in d0["macro_w"][i].tolist()]
        ci = d0["cluster_idx"][i].tolist()
        print(f"  [{i}] {t!r:20s} 天王{mi}:{mw} 宗门{[CLUSTER_NAMES[c] for c in ci]}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="写一个Python快排函数", help="用户问题")
    ap.add_argument("--domain", type=int, default=0, help="期望core_id 0-7, -1=不判定")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--probe-only", action="store_true", help="只探测不生成")
    ap.add_argument("--weight", default="myriad_moe_hierarchical_weights_v3.pt")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit"])
    ap.add_argument("--top-tokens", type=int, default=8)
    ap.add_argument("--base-only", action="store_true", help="关掉MoE增量(gamma=0), 看纯基座输出, 定位质量问题归谁")
    a = ap.parse_args()

    model, tok, wrappers, _proc = load_system(weight_path=a.weight, quant=a.quant)
    if a.base_only:
        for w in wrappers:
            w.gamma = 0.0
        print("[*] base-only模式: MoE增量已关闭")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    prompt_fmt = f"<start_of_turn>user\n{a.prompt}<end_of_turn>\n<start_of_turn>model\n"
    ids = tok.encode(prompt_fmt, return_tensors="pt").to("cuda:0")

    if a.probe_only:
        diags = probe_forward(model, wrappers, ids, pad)
        report(diags, tok, ids, expect_core=a.domain if a.domain >= 0 else None, top_tokens=a.top_tokens)
        return

    print("[*] 生成中...")
    with torch.no_grad():
        from transformers.generation import StoppingCriteriaList, StopStringCriteria
        out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                             repetition_penalty=1.1,
                             stopping_criteria=StoppingCriteriaList(
                                 [StopStringCriteria(tokenizer=tok, stop_strings=["<end_of_turn>"])]),
                             pad_token_id=pad, use_cache=True)
    new_toks = out[0, ids.shape[1]:]
    import re
    text = re.split(r"<end_of_turn>?", tok.decode(new_toks, skip_special_tokens=False))[0]
    print("\n[模型输出]\n" + text.strip() + "\n")

    diags = probe_forward(model, wrappers, out[:, :ids.shape[1] + len(new_toks)], pad)
    report(diags, tok, out, expect_core=a.domain if a.domain >= 0 else None, top_tokens=a.top_tokens)


if __name__ == "__main__":
    main()
