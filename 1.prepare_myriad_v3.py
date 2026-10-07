# ==============================================================================
# 1.prepare_myriad_v3.py -- Myriad-MoE v3 全新数据管线 (28k, 8x3500, 16x1750)
# 相对 v2.4 的升级:
#   [V3-1] core5/6/7 不再共用alpaca一锅粥, 各有专属源:
#          5逻辑=facebook/natural_reasoning (Llama-70B推理链),
#          6约束=alpaca[:20000]大池定额取最像约束的3500 (池子大7倍, 尾部更纯),
#          7中文特区=Belle 0.5M CN (实际使用语言! 插槽A/B改为中文创作/中文问答)
#   [V3-2] 宗门13 负向安全拿到真实拒答数据: mlabonne/harmful_behaviors 416条 + alpaca否定约束补齐
#   [V3-3] 每core 3500 (no_robots用量重算后仍有富余), 定额平衡, 断言锁死
# 输出: myriad_train_data_v3.jsonl + myriad_tokenized_cache_v3.pt (schema与v2兼容)
# 注意: 全新标签体系, 必须从step 0训练, 用 checkpoint_v3.pt / 权重_v3 (训练脚本DATA_TAG=v3)
# ==============================================================================
import os
# 不设离线: 缓存命中走缓存, 新数据(Belle/natural_reasoning)走直连下载
# (本机代理变量会搞坏datasets streaming, 运行前 unset 各类 *PROXY, 见 train 脚本注释)
for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

import ast
import json
import re
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

CORE_N, CLU_N = 3500, 1750
print("=" * 80)
print("正在装配【Myriad-MoE v3 数据】(8x3500, 16x1750, 专属源+中文特区+真实拒答)...")
print("=" * 80)


def score(text, patterns):
    text = text.lower()
    return sum(len(re.findall(p, text)) for p in patterns)


def topk(items, k, key):
    ranked = sorted(enumerate(items), key=lambda t: (-key(t[1]), t[0]))
    return [items[i] for i, _ in ranked[:k]], [items[i] for i, _ in ranked[k:]]


def take(items, k, name):
    assert len(items) >= k, f"[✗] {name} 池子不够: {len(items)} < {k}"
    return items[:k]


C_PAT = [r'json', r'table', r'bullet', r'translat', r'grammar', r'summar', r'rewrit',
         r'classif', r'extract', r'email', r'letter', r'format', r'alphabet', r'convert',
         r'acronym', r'synonym', r'antonym', r'plural', r'tense', r'punctuat', r'spell',
         r'categor', r'title', r'headline', r'keyword', r'outline']
L_PAT = [r'\bwhy\b', r'reason', r'cause', r'because', r'explain', r'compar', r'contrast',
         r'difference', r'analy', r'\bsteps?\b', r'logic', r'prov', r'conclud', r'infer',
         r'deduc', r'pros and cons', r'argu', r'opinion', r'moral', r'philosoph',
         r'riddle', r'puzzle', r'paradox', r'assum', r'hypoth', r'what if']
ALG_PAT = [r'sort', r'search', r'recursi', r'fibonacci', r'factorial', r'dynamic',
           r'\btree\b', r'graph', r'binary', r'quick', r'bubble', r'algorithm',
           r'palindrome', r'prime', r'knapsack', r'travers']
EXP_PAT = [r'experiment', r'observ', r'measur', r'test', r'hypothes', r'evidence', r'result']
FMT_PAT = [r'json', r'table', r'bullet', r'\blist\b', r'format', r'alphabet']
NEG_PAT = [r"do not", r"don't", r'never', r'avoid', r'must not', r'refuse', r'without',
           r'prohibit', r'forbidden', r'not allowed', r'no more than', r'at most',
           r'only \d+', r'cannot', r"shouldn't"]
CAUSE_PAT = [r'\bwhy\b', r'cause', r'because', r'\bsteps?\b', r'\bhow\b', r'explain']
CN_CREATE_PAT = [r'写', r'作诗', r'故事', r'创作', r'编', r'小说', r'广告', r'标题', r'作文']

REFUSALS = [
    "I can't help with that because it could facilitate wrongdoing or harm.",
    "Sorry, but I can't assist with this request. If you have a lawful question instead, I'm glad to help.",
    "I'm not able to comply with this request since it involves potentially harmful activity.",
    "I have to refuse this request. Let me know if there's something else I can help with instead.",
]

myriad_data = []

# --- Core 0 代码 (3500) ---
print("    - [1/8] Core 0 代码工程...")
ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:3500]")
items = []
for it in ds:
    p = it["instruction"] + (f"\n{it['input']}" if it.get("input") else "")
    items.append({"prompt": p, "response": it["output"], "s": score(p + "\n" + it["output"], ALG_PAT)})
a, b = topk(items, CLU_N, lambda x: x["s"])
for x in a:
    myriad_data.append({"core_id": 0, "core_name": "Code", "cluster_id": 0, "prompt": x["prompt"], "response": x["response"]})
for x in b:
    myriad_data.append({"core_id": 0, "core_name": "Code", "cluster_id": 1, "prompt": x["prompt"], "response": x["response"]})

# --- Core 1 数学 (3500) ---
print("    - [2/8] Core 1 严密数学...")
ds = load_dataset("openai/gsm8k", "main", split="train[:3500]")
items = [{"prompt": it["question"], "response": it["answer"], "s": it["answer"].count("<<")} for it in ds]
a, b = topk(items, CLU_N, lambda x: x["s"])
for x in a:
    myriad_data.append({"core_id": 1, "core_name": "Math", "cluster_id": 2, "prompt": x["prompt"], "response": x["response"]})
for x in b:
    myriad_data.append({"core_id": 1, "core_name": "Math", "cluster_id": 3, "prompt": x["prompt"], "response": x["response"]})

# --- Core 2 科学 (3500) ---
print("    - [3/8] Core 2 自然科学...")
ds = load_dataset("allenai/sciq", split="train[:3500]")
items = []
for it in ds:
    p = f"Question: {it['question']}\nContext: {it['support']}"
    items.append({"prompt": p, "response": it["correct_answer"],
                  "s": score(it["question"] + "\n" + it["support"], EXP_PAT)})
a, b = topk(items, CLU_N, lambda x: x["s"])
for x in b:  # 注意: a=实验在后, 保持 cluster 4=物化 / 5=实验 的旧编号
    myriad_data.append({"core_id": 2, "core_name": "Science", "cluster_id": 4, "prompt": x["prompt"], "response": x["response"]})
for x in a:
    myriad_data.append({"core_id": 2, "core_name": "Science", "cluster_id": 5, "prompt": x["prompt"], "response": x["response"]})

# --- Core 3/4 no_robots (各3500) ---
print("    - [4/8][5/8] Core 3/4 (category分流)...")
ds = load_dataset("HuggingFaceH4/no_robots", split="train")
buckets = {}
for it in ds:
    msgs = it["messages"]
    if len(msgs) < 2 or not msgs[1]["content"].strip():
        continue
    buckets.setdefault(it.get("category", "?"), []).append({"prompt": msgs[0]["content"], "response": msgs[1]["content"]})
gen, brs = buckets.get("Generation", []), buckets.get("Brainstorm", [])
gen_desc = sorted(gen, key=lambda x: len(x["response"]), reverse=True)
c6 = take(gen_desc, CLU_N, "长篇叙事")
used = set(map(id, c6))
c7 = take(brs + [x for x in sorted(gen, key=lambda x: len(x["response"])) if id(x) not in used], CLU_N, "诗歌意象")
for x in c6:
    myriad_data.append({"core_id": 3, "core_name": "Creative_Arts", "cluster_id": 6, "prompt": x["prompt"], "response": x["response"]})
for x in c7:
    myriad_data.append({"core_id": 3, "core_name": "Creative_Arts", "cluster_id": 7, "prompt": x["prompt"], "response": x["response"]})
qa_pool = buckets.get("Open QA", []) + buckets.get("Closed QA", []) + buckets.get("Chat", [])
sum_pool = buckets.get("Summarize", []) + buckets.get("Rewrite", []) + buckets.get("Classify", []) + buckets.get("Extract", [])
c8 = take(sorted(qa_pool, key=lambda x: len(x["response"]), reverse=True), CLU_N, "专业问答")
c9 = sum_pool[:CLU_N]
if len(c9) < CLU_N:
    have = set(map(id, c8)) | set(map(id, c9))
    fill = [x for x in sorted(buckets.get("Chat", []), key=lambda x: len(x["response"])) if id(x) not in have]
    need = CLU_N - len(c9)
    c9 = c9 + take(fill, need, "Chat补齐")
    print(f"      [!] 摘要类短{need}条, 用短Chat补齐")
for x in c8:
    myriad_data.append({"core_id": 4, "core_name": "Business_Dialogue", "cluster_id": 8, "prompt": x["prompt"], "response": x["response"]})
for x in c9:
    myriad_data.append({"core_id": 4, "core_name": "Business_Dialogue", "cluster_id": 9, "prompt": x["prompt"], "response": x["response"]})

# --- Core 5 逻辑: alpaca大池定额取最像逻辑的3500 ---
# (备选 facebook/natural_reasoning 流式被hub未登录限流, 有HF_TOKEN后再升级)
print("    - [6/8] Core 5 逻辑思辨 (alpaca大池)...")
ds = load_dataset("tatsu-lab/alpaca", split="train[20000:50000]")
pool5 = []
for it in ds:
    p = it["instruction"] + (f"\n{it['input']}" if it.get("input") else "")
    if not it["output"].strip():
        continue
    pool5.append({"prompt": p, "response": it["output"],
                  "s": score(p + "\n" + it["output"], L_PAT)})
print(f"      逻辑命中{sum(1 for x in pool5 if x['s']>0)}/{len(pool5)}")
logi5, _ = topk(pool5, CORE_N, lambda x: x["s"])
a, b = topk(logi5, CLU_N, lambda x: score(x["prompt"], CAUSE_PAT))
for x in a:  # 10 因果推演
    myriad_data.append({"core_id": 5, "core_name": "Logic_Philosophy", "cluster_id": 10, "prompt": x["prompt"], "response": x["response"]})
for x in b:  # 11 哲学批判
    myriad_data.append({"core_id": 5, "core_name": "Logic_Philosophy", "cluster_id": 11, "prompt": x["prompt"], "response": x["response"]})

# --- Core 6 约束: alpaca大池定额取最像约束的3500 ---
print("    - [7/8] Core 6 严格约束 (alpaca大池)...")
ds = load_dataset("tatsu-lab/alpaca", split="train[:20000]")
pool = []
for it in ds:
    p = it["instruction"] + (f"\n{it['input']}" if it.get("input") else "")
    if not it["output"].strip():
        continue
    pool.append({"prompt": p, "response": it["output"], "s": score(p + "\n" + it["output"], C_PAT)})
print(f"      约束命中{sum(1 for x in pool if x['s']>0)}/{len(pool)}")
cons, _ = topk(pool, CORE_N, lambda x: x["s"])
f1, f2 = topk(cons, CLU_N, lambda x: score(x["prompt"], FMT_PAT) - score(x["prompt"], NEG_PAT))
for x in f1:  # 12 格式规范
    myriad_data.append({"core_id": 6, "core_name": "Constraint_Rules", "cluster_id": 12, "prompt": x["prompt"], "response": x["response"]})
neg_pool = f2  # 13 一半真实拒答 + 一半否定约束, 下面合并

# --- Core 7 中文特区: Belle流式取3500 ---
print("    - [8/8] Core 7 中文特区 (Belle)...")
ds = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
zh_items = []
for it in ds:
    p = it["instruction"] + (f"\n{it['input']}" if it.get("input") else "")
    r = it["output"]
    if not r or not r.strip() or len(r) < 5:
        continue
    zh_items.append({"prompt": p, "response": r.strip()})
    if len(zh_items) >= CORE_N:
        break
assert len(zh_items) == CORE_N, f"中文池不够: {len(zh_items)}"
a, b = topk(zh_items, CLU_N, lambda x: score(x["prompt"], CN_CREATE_PAT))
for x in a:  # 14 中文创作
    myriad_data.append({"core_id": 7, "core_name": "Chinese_Slots", "cluster_id": 14, "prompt": x["prompt"], "response": x["response"]})
for x in b:  # 15 中文问答
    myriad_data.append({"core_id": 7, "core_name": "Chinese_Slots", "cluster_id": 15, "prompt": x["prompt"], "response": x["response"]})

# --- 宗门13: 真实拒答416 + 否定约束补齐 ---
print("    - [拒答] 宗门13 负向安全 (harmful + 否定约束)...")
ds = load_dataset("mlabonne/harmful_behaviors", split="train")
ref = [{"prompt": it["text"], "response": REFUSALS[i % len(REFUSALS)]} for i, it in enumerate(ds)]
print(f"      真实有害请求 {len(ref)} 条配固定拒答模板")
neg_ranked = sorted(neg_pool, key=lambda x: score(x["prompt"], NEG_PAT), reverse=True)
c13 = (ref + neg_ranked)[:CLU_N]
assert len(c13) == CLU_N, f"宗门13不够: {len(c13)}"
# core6共3500 = f1(1750, cluster12) + c13(1750, cluster13: 416真实拒答 + 1334否定约束), 互斥
core6_neg = [x for x in c13 if x["response"] not in REFUSALS]
print(f"      宗门13构成: 真实拒答{sum(1 for x in c13 if x['response'] in REFUSALS)} + 否定约束{len(core6_neg)}")
for x in c13:
    myriad_data.append({"core_id": 6, "core_name": "Constraint_Rules", "cluster_id": 13, "prompt": x["prompt"], "response": x["response"]})

# ----------------------------------------------------------------------
# 断言 + 报告
# ----------------------------------------------------------------------
assert len(myriad_data) == 28000, f"总数 {len(myriad_data)} != 28000"
assert sorted(Counter(r["core_id"] for r in myriad_data).values()) == [CORE_N] * 8, "core不平衡"
assert sorted(Counter(r["cluster_id"] for r in myriad_data).values()) == [CLU_N] * 16, "cluster不平衡"
print(f"\n[✔] 组装完毕: {len(myriad_data)}条, 8 core各{CORE_N}, 16 cluster各{CLU_N}")
for cid in range(16):
    r = next(x for x in myriad_data if x["cluster_id"] == cid)
    print(f"    宗门{cid}: core{r['core_id']} P:{r['prompt'][:60].replace(chr(10),' ')}...")

with open("myriad_train_data_v3.jsonl", "w", encoding="utf-8") as f:
    for entry in myriad_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("[✔] JSONL 完成")

model_id = "../gemma-4-12B-it-qat-q4_0-unquantized"
print("[*] Tokenizer 离线压缩...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
MAX_LENGTH = 448
tokenized_records = []
for item in myriad_data:
    user_prompt = f"<start_of_turn>user\n{item['prompt']}<end_of_turn>\n<start_of_turn>model\n"
    full_text = f"{user_prompt}{item['response']}<end_of_turn>"
    prompt_ids = tokenizer.encode(user_prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_ids) > MAX_LENGTH:
        full_ids = full_ids[:MAX_LENGTH]
    input_ids = torch.tensor(full_ids, dtype=torch.int32)
    labels = input_ids.clone()
    labels[:min(len(prompt_ids), len(labels))] = -100
    tokenized_records.append({"input_ids": input_ids, "labels": labels,
                              "core_id": item["core_id"], "cluster_id": item["cluster_id"],
                              "length": len(full_ids)})
tokenized_records.sort(key=lambda x: x["length"])
torch.save(tokenized_records, "myriad_tokenized_cache_v3.pt")
print(f"[✔] 缓存完成: {len(tokenized_records)}条")
print("=" * 80)
