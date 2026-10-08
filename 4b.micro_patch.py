#!/usr/bin/env python3
# ==============================================================================
# 4b.micro_patch.py -- 宗门切片定向补丁 (细粒度知识 hotfix)
# 只训目标宗门的 micro_lora 切片 (每宗门每层约2M参数, 1宗门bf16约48MB)
# + router_cluster整表 (小) + router_micro目标行; 其余切片梯度清零, 不漂移。
# --clusters 不给时从数据里的 cluster_id 自动推导 (>4个宗门会警告包变大)。
# 产物走 /api/hotswap 切片合并, 多宗门补丁可 sequential 叠加。
# 用法: python3 4b.micro_patch.py --data poetry_patch.jsonl --out patch_micro_poetry.pt
#       python3 4b.micro_patch.py --data sci_patch.jsonl --clusters 4,5 --out patch_micro_sci.pt
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
trainmod = patch_lib.trainmod


def make_clan_mask(clans):
    """非目标宗门切片/行梯度清零 (Adam动量下零梯度≈不动, 冻结切片不漂移)."""
    clans = sorted({int(c) for c in clans})
    for c in clans:
        assert 0 <= c < 16, f"宗门下标越界: {c}"

    def _mask(wraps):
        for w in wraps:
            per = w.experts_per_cluster * w.micro_rank
            keep = torch.zeros(w.total_micro_rank, dtype=torch.bool)
            rows = []
            for c in clans:
                keep[c * per:(c + 1) * per] = True
                rows += list(range(c * w.experts_per_cluster,
                                  (c + 1) * w.experts_per_cluster))
            if w.micro_lora_A.grad is not None:
                w.micro_lora_A.grad[:, ~keep] = 0
            if w.micro_lora_B.grad is not None:
                w.micro_lora_B.grad[~keep, :] = 0
            rg = w.router_micro.weight.grad
            if rg is not None:
                drop = torch.ones(rg.shape[0], dtype=torch.bool)
                drop[rows] = False
                rg[drop, :] = 0
    _mask.clans = clans
    return _mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="poetry_patch.jsonl")
    ap.add_argument("--base-weights", default="myriad_moe_hierarchical_weights_v3.pt")
    ap.add_argument("--out", default="patch_micro.pt")
    ap.add_argument("--model-id", default="../gemma-4-12B-it-qat-q4_0-unquantized")
    ap.add_argument("--clusters", default="",
                    help="目标宗门, 逗号分隔 (如4,5); 不给则从数据自动推导")
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
    recs, auto_clusters = patch_lib.load_patch_data(tok, a.data)
    clans = sorted({int(x) for x in a.clusters.split(",") if x.strip() != ""}) \
        or auto_clusters
    if len(clans) > 4:
        print(f"[!] 目标宗门{len(clans)}个, 包约{len(clans) * 48:.0f}MB; "
              f"建议按宗门拆数据分开打补丁 (可hotswap叠加)")
    print(f"[*] 目标宗门: {clans}")

    # --- 解冻: 目标切片 (A/B整表解冻 + 梯度掩码) + 宗门门控整表 + micro门控 (行掩码) ---
    for w in wraps:
        w.micro_lora_A.requires_grad_(True)
        w.micro_lora_B.requires_grad_(True)
        for p_ in w.router_cluster.router.parameters():
            p_.requires_grad = True
        for p_ in w.router_micro.parameters():
            p_.requires_grad = True

    mask = make_clan_mask(clans)
    patch_lib.run_patch_loop(model, wraps, recs, pad, device, a.lr, a.epochs,
                             a.batch, a.accum, grad_mask_fn=mask, tag="micro")

    wd = {}
    for i, w in enumerate(wraps):
        idx = 18 + i
        wd[f"layer_{idx}_router_cluster"] = {
            k: v.data.cpu() for k, v in w.router_cluster.router.state_dict().items()}
        for c in clans:
            s0, s1, r0, r1 = trainmod._clan_slices(w, c)
            wd[f"layer_{idx}_micro_clan_{c}_A"] = w.micro_lora_A.data[:, s0:s1].cpu()
            wd[f"layer_{idx}_micro_clan_{c}_B"] = w.micro_lora_B.data[s0:s1, :].cpu()
            wd[f"layer_{idx}_router_micro_clan_{c}"] = \
                w.router_micro.weight.data[r0:r1, :].cpu()
    print(f"[*] 宗门包约 {patch_lib.patch_size_mb(wd):.1f} MB (全量约810MB)")
    patch_lib.save_patch(a.out, "micro", clans, wd)


if __name__ == "__main__":
    main()
