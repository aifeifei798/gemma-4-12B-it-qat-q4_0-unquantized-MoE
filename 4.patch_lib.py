#!/usr/bin/env python3
# ==============================================================================
# 4.patch_lib.py -- 拆分补丁公共件 (4a天王 / 4b宗门共用, 不直接运行)
# 数据模板、装机流程、训练循环与 4.micro_patch.py 完全一致, 只是可冻结/可掩码。
# ==============================================================================
import gc
import importlib.util
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "2.train_myriad_v2_24g_safe.py")
_spec = importlib.util.spec_from_file_location("trainmod", _train_path)
trainmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainmod)


def load_patch_data(tok, data_path):
    """同训练模板 (response-only); 返回 (recs, clusters去重排序)."""
    recs = []
    for line in open(data_path, encoding="utf-8"):
        it = json.loads(line)
        up = f"<start_of_turn>user\n{it['prompt']}<end_of_turn>\n<start_of_turn>model\n"
        full = f"{up}{it['response']}<end_of_turn>"
        pids, fids = tok.encode(up, add_special_tokens=False), tok.encode(full, add_special_tokens=False)
        fids = fids[:448]
        ii = torch.tensor(fids, dtype=torch.int32)
        lb = ii.clone()
        lb[:min(len(pids), len(lb))] = -100
        recs.append({"input_ids": ii, "labels": lb, "core_id": it.get("core_id", 7),
                      "cluster_id": it.get("cluster_id", 15), "length": len(fids)})
    clusters = sorted({r["cluster_id"] for r in recs})
    print(f"[*] 补丁样本 {len(recs)} 条, 涉及宗门 {clusters}")
    return recs, clusters


def build_patch_system(model_id, base_weights, quant, device, dtype):
    """基座常驻量化 + 12层wrapper + v3权重 + 全冻结. 调用方再按需解冻."""
    from transformers import BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    qc = (BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0) if quant == "8bit"
          else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=qc,
                                                device_map=device, attn_implementation="sdpa")
    for attr in ("embed_vision", "embed_audio"):
        if hasattr(model.model, attr):
            delattr(model.model, attr)
    gc.collect(); torch.cuda.empty_cache()
    for p_ in model.parameters():
        p_.requires_grad = False
    tm = model.model.language_model
    hidden = getattr(model.config.text_config, "hidden_size", 3840)
    wraps = []
    for idx in range(18, 30):
        w = trainmod.MyriadTrueRoutingLayer(tm.layers[idx].mlp, hidden_dim=hidden,
                                            device=device, dtype=dtype)
        tm.layers[idx].mlp = w
        wraps.append(w)
    sd = torch.load(base_weights, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "weights" in sd and "scope" in sd:
        raise ValueError("[!] 底座不能是拆分补丁, 请用v3全量权重 (拆分补丁只走hotswap叠加)")
    trainmod.load_weights_dict(wraps, 18, sd)
    print("[*] v3权重已载入, 非目标参数保持冻结...")
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    return model, tok, pad, wraps


def run_patch_loop(model, wraps, recs, pad, device, lr, epochs, batch, accum,
                   grad_mask_fn=None, tag="patch"):
    """训练循环 (与4.micro_patch一致); grad_mask_fn(wraps)在clip前把非目标梯度清零."""
    import bitsandbytes as bnb
    params = [p_ for w in wraps for p_ in w.parameters() if p_.requires_grad]
    print(f"[*] 可训参数 {sum(p.numel() for p in params) / 1e6:.1f}M "
          f"(全量{sum(p.numel() for w in wraps for p in w.parameters()) / 1e6:.1f}M)")
    opt = bnb.optim.PagedAdamW8bit(params, lr=lr)
    dl = DataLoader(recs, batch_size=batch, shuffle=True,
                    generator=torch.Generator().manual_seed(7),
                    num_workers=0,
                    collate_fn=lambda b: trainmod.dynamic_collate_fn(b, pad_token_id=pad))
    model.eval()
    for w in wraps:
        w.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        for b in dl:
            ids = b["input_ids"].to(device)
            lab = b["labels"].to(device)
            core = b["core_ids"].to(device)
            clu = b["cluster_ids"].to(device)
            lm, aux, sup, am, ac, S, n = trainmod.forward_backward_batch(
                model, wraps, 12, ids, lab, core, clu, pad,
                0.005, 0.3, 1.0 / accum, max_tokens=256, sup_response_only=True)
            if (step + 1) % accum == 0:
                if grad_mask_fn is not None:
                    grad_mask_fn(wraps)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
            step += 1
        print(f"  [{tag} epoch{ep}]: LM {lm.item():.4f} aux {aux.item():.4f} "
              f"sup {sup.item():.4f} acc {am.item():.1%} ({time.time() - t0:.1f}s)")
        t0 = time.time()


def save_patch(out, scope, clusters, weights):
    torch.save({"scope": scope, "clusters": list(clusters), "base": "v3",
                "weights": weights}, out)
    print(f"[✔] 补丁已存: {out} (scope={scope}, 可热插拔, 服务端无需重启)")


def patch_size_mb(weights):
    tot = 0
    for v in weights.values():
        if torch.is_tensor(v):
            tot += v.numel() * v.element_size()
        elif isinstance(v, dict):  # router state_dict
            tot += sum(t.numel() * t.element_size() for t in v.values()
                       if torch.is_tensor(t))
    return tot / 1e6
