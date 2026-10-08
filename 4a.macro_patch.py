#!/usr/bin/env python3
# ==============================================================================
# 4a.macro_patch.py -- 天王层定向补丁 (领域路由手感修正)
# 只训 macro_lora_A/B + router_macro (12层约12M参数, bf16约24MB),
# shared/micro/宗门门控全程冻结。产物走 /api/hotswap 切片合并, 可多补丁叠加。
# 用法: python3 4a.macro_patch.py --data poetry_patch.jsonl --out patch_macro_poetry.pt
#       (旧 4.micro_patch.py 全量流程保留, 当回退用)
# ==============================================================================
import argparse
import importlib.util
import os

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_lib = importlib.util.spec_from_file_location(
    "patch_lib", os.path.join(os.path.dirname(os.path.abspath(__file__)), "4.patch_lib.py"))
patch_lib = importlib.util.module_from_spec(_lib)
_lib.loader.exec_module(patch_lib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="poetry_patch.jsonl")
    ap.add_argument("--base-weights", default="myriad_moe_hierarchical_weights_v3.pt")
    ap.add_argument("--out", default="patch_macro.pt")
    ap.add_argument("--model-id", default="../gemma-4-12B-it-qat-q4_0-unquantized")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit"])
    a = ap.parse_args()

    device, dtype = "cuda:0", torch.bfloat16
    torch.manual_seed(42)

    model, tok, pad, wraps = patch_lib.build_patch_system(
        a.model_id, a.base_weights, a.quant, device, dtype)
    recs, _ = patch_lib.load_patch_data(tok, a.data)

    # --- 只解冻天王分支 ---
    for w in wraps:
        w.macro_lora_A.requires_grad_(True)
        w.macro_lora_B.requires_grad_(True)
        for p_ in w.router_macro.router.parameters():
            p_.requires_grad = True

    patch_lib.run_patch_loop(model, wraps, recs, pad, device, a.lr, a.epochs,
                             a.batch, a.accum, grad_mask_fn=None, tag="macro")

    wd = {}
    for i, w in enumerate(wraps):
        idx = 18 + i
        wd[f"layer_{idx}_macro_lora_A"] = w.macro_lora_A.data.cpu()
        wd[f"layer_{idx}_macro_lora_B"] = w.macro_lora_B.data.cpu()
        wd[f"layer_{idx}_router_macro"] = {
            k: v.data.cpu() for k, v in w.router_macro.router.state_dict().items()}
    print(f"[*] 天王包约 {patch_lib.patch_size_mb(wd):.1f} MB (全量约810MB)")
    patch_lib.save_patch(a.out, "macro", [], wd)


if __name__ == "__main__":
    main()
