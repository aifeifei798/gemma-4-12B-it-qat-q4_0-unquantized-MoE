#!/usr/bin/env python3
# ==============================================================================
# 4.micro_patch.py -- 定向微补丁训练 (例: 诗歌翻车 hotfix)
# 复用 2.train 脚本的 forward_backward_batch (response-only监督 + aux),
# 在已训好的 v3 权重上, 用几十条定向样本小步快训, 产出可热插拔权重文件。
# 用法: python3 4.micro_patch.py --data poetry_patch.jsonl --out myriad_moe_patch_poetry.pt
# ==============================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, gc, json, time, torch
import importlib.util

spec = importlib.util.spec_from_file_location("trainmod", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "2.train_myriad_v2_24g_safe.py"))
trainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainmod)
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="poetry_patch.jsonl")
    ap.add_argument("--base-weights", default="myriad_moe_hierarchical_weights_v3.pt")
    ap.add_argument("--out", default="myriad_moe_patch_poetry.pt")
    ap.add_argument("--model-id", default="../gemma-4-12B-it-qat-q4_0-unquantized")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit"])
    a = ap.parse_args()

    device, dtype = "cuda:0", torch.bfloat16
    torch.manual_seed(42)

    # --- 数据 (同训练模板) ---
    tok = AutoTokenizer.from_pretrained(a.model_id)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    recs = []
    for line in open(a.data, encoding="utf-8"):
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
    print(f"[*] 补丁样本 {len(recs)} 条")

    # --- 模型 + v3权重 ---
    from transformers import BitsAndBytesConfig
    qc = (BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0) if a.quant == "8bit"
          else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True))
    model = AutoModelForCausalLM.from_pretrained(a.model_id, quantization_config=qc,
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
        w = trainmod.MyriadTrueRoutingLayer(tm.layers[idx].mlp, hidden_dim=hidden, device=device, dtype=dtype)
        tm.layers[idx].mlp = w
        wraps.append(w)
    sd = torch.load(a.base_weights, map_location="cpu", weights_only=False)
    trainmod.load_weights_dict(wraps, 18, sd)
    print("[*] v3权重已载入, 开始微训...")
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    import bitsandbytes as bnb
    params = [p_ for w in wraps for p_ in w.parameters() if p_.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=a.lr)
    dl = DataLoader(recs, batch_size=a.batch, shuffle=True,
                    generator=torch.Generator().manual_seed(7),
                    num_workers=0,
                    collate_fn=lambda b: trainmod.dynamic_collate_fn(b, pad_token_id=pad))
    model.eval()
    for w in wraps:
        w.train()
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        for batch in dl:
            ids = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            core = batch["core_ids"].to(device)
            clu = batch["cluster_ids"].to(device)
            lm, aux, sup, am, ac, S, n = trainmod.forward_backward_batch(
                model, wraps, 12, ids, lab, core, clu, pad,
                0.005, 0.3, 1.0 / a.accum, max_tokens=256, sup_response_only=True)
            if (step + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
            step += 1
        print(f"  epoch{ep}: LM {lm.item():.4f} aux {aux.item():.4f} sup {sup.item():.4f} "
              f"acc {am.item():.1%} ({time.time() - t0:.1f}s)")
        t0 = time.time()

    out = trainmod.extract_weights_dict(wraps, 18)
    torch.save(out, a.out)
    print(f"[✔] 补丁已存: {a.out} (可热插拔, 服务端无需重启)")


if __name__ == "__main__":
    main()
