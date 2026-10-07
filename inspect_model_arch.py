import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "../gemma-4-12B-it-qat-q4_0-unquantized"

print("=" * 75)
print(f"🔍 正在对模型结构进行元拓扑透视: {model_id}")
print("=" * 75)

# 1. 尝试以 meta 模式秒速实例化结构骨架 (零显存、零权重等待)
try:
    config = AutoConfig.from_pretrained(model_id)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    print("[✔] 成功在 meta 设备上构建模型骨架！")
except Exception as e:
    print(f"[!] meta 模式加载失败 ({e})，回退到常规轻量加载模式...")
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="cpu",
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.bfloat16)

print("\n" + "=" * 30 + " [1. 顶层子模块树] " + "=" * 30)
for name, child in model.named_children():
    print(f"  ├── {name}  -->  {child.__class__.__name__}")
    # 往下深入探一层
    for sub_name, sub_child in child.named_children():
        print(f"  │    ├── {sub_name}  -->  {sub_child.__class__.__name__}")

print("\n" + "=" * 30 + " [2. 关键属性与 Layers 探测] " + "=" * 30)
candidate_paths = [
    "model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "text_model.layers",
    "text_model.encoder.layers",
    "model.text_model.layers",
    "model.language_model.model.layers",
    "model.model.layers",
]

found_layers = None
found_path = None

for path in candidate_paths:
    curr = model
    success = True
    for part in path.split("."):
        if hasattr(curr, part):
            curr = getattr(curr, part)
        else:
            success = False
            break
    if success and isinstance(curr, (nn.ModuleList, list)):
        found_layers = curr
        found_path = path
        print(f"  [🎯 精准命中] 发现层列表路径: model.{path}")
        print(f"      - 包含层数: {len(found_layers)}")
        break

if not found_layers:
    print("  [!] 常用候选路径均未直接命中，正在全局递归搜索 ModuleList 结构...")
    for full_name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 10:
            print(
                f"  [💡 发现疑似主干 Layers]: model.{full_name} (长度: {len(module)}, 元素类型: {module[0].__class__.__name__})"
            )
            if found_layers is None:
                found_layers = module
                found_path = full_name

# 3. 深入探测单个 Layer 内部结构 (重点是 MLP 命名)
if found_layers is not None:
    sample_layer = found_layers[0]
    print("\n" + "=" * 30 + " [3. 单层 Block 内部结构 (第 0 层)] " + "=" * 30)
    print(f"Layer 0 类名: {sample_layer.__class__.__name__}\n子模块组成:")
    for child_name, child_module in sample_layer.named_children():
        print(f"  ├── {child_name}  -->  {child_module.__class__.__name__}")

    print("\n" + "=" * 30 + " [4. 建议的适配修改代码] " + "=" * 30)
    mlp_candidates = [
        k for k, _ in sample_layer.named_children()
        if any(x in k.lower() for x in ["mlp", "ffn", "feed_forward"])
    ]
    print(f"提取 Layers 代码: layers = model.{found_path}")
    if mlp_candidates:
        print(
            f"嫁接替换目标:     layer.{mlp_candidates[0]} = MyriadLayerWrapper(layer.{mlp_candidates[0]}, ...)"
        )
    else:
        print(f"未自动识别出 mlp/ffn 命名，请观察第 3 部分的子模块输出！")
else:
    print("\n[❌] 未能自动定位 layers，打印模型所有一级与二级属性字典供人工排查:")
    print(dir(model))

print("\n" + "=" * 75)
