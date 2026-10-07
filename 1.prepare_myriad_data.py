# ==============================================================================
# 1.prepare_myriad_data.py v2.4 [语义分簇版]:
# 相对 v2.2 的问题与修复:
#   [D1] 旧版宗门按 idx%2 交替划分 = 随机标签, cluster监督学的是噪声
#        -> 改为每core内语义打分排序, 定额取前1500 (天然平衡 + 语义最纯)
#   [D2] core 5/6/7 全是同一alpaca池子的任意切片, 内容不可分 (" stayed healthy" 凭什么是逻辑?)
#        -> 改为关键词打分排序定额三分: 约束(6) / 逻辑(5) / 插槽(7)
#   [D3] no_robots 的 category 字段旧版完全没用
#        -> core3=Generation+Brainstorm, core4=问答摘要类, Coding剔除(与core0撞车)
#   [D4] 5条空response旧版直接入库
#        -> 过滤 + 数量断言 (core恒3000, cluster恒1500)
# 输出(与训练脚本兼容, schema不变): myriad_train_data_v2.jsonl + myriad_tokenized_cache_v2.pt
# 注意: 标签语义变了, 旧断点/旧权重全部作废, 必须从step 0重训!
# ==============================================================================
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 只用本地缓存, 可复现
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import re
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

print("=" * 80)
print("正在装配【Myriad-MoE v2.4: 语义分簇语料库】(8天王 x 16宗门, 定额平衡)...")
print("=" * 80)


def score(text, patterns):
    text = text.lower()
    return sum(len(re.findall(p, text)) for p in patterns)


def topk(items, k, key):
    """按key降序稳定取前k个,  ties按原顺序 (确定性). key返回数值越大越优先."""
    ranked = sorted(enumerate(items), key=lambda t: (-key(t[1]), t[0]))
    return [items[i] for i, _ in ranked[:k]], [items[i] for i, _ in ranked[k:]]


def take(items, k, name):
    assert len(items) >= k, f"[✗] {name} 池子不够: {len(items)} < {k}"
    return items[:k]


# ----------------------------------------------------------------------
# 关键词表 (打分用, 命中越多越像该类)
# ----------------------------------------------------------------------
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

myriad_data = []

# ----------------------------------------------------------------------
# Core 0: 代码工程 (3000)
# ----------------------------------------------------------------------
print("    - [1/8] Core 0 代码工程...")
ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:3000]")
code_items = []
for item in ds_code:
    p = item["instruction"] + (f"\n{item['input']}" if item.get("input") else "")
    code_items.append({"prompt": p, "response": item["output"],
                       "s": score(p + "\n" + item["output"], ALG_PAT)})
alg, eng = topk(code_items, 1500, lambda x: x["s"])   # 0 语法算法 vs 1 系统工程
for x in alg:
    myriad_data.append({"core_id": 0, "core_name": "Code", "cluster_id": 0, "prompt": x["prompt"], "response": x["response"]})
for x in eng:
    myriad_data.append({"core_id": 0, "core_name": "Code", "cluster_id": 1, "prompt": x["prompt"], "response": x["response"]})

# ----------------------------------------------------------------------
# Core 1: 严密数学 (3000)
# ----------------------------------------------------------------------
print("    - [2/8] Core 1 严密数学...")
ds_math = load_dataset("openai/gsm8k", "main", split="train[:3000]")
math_items = [{"prompt": it["question"], "response": it["answer"],
               "s": it["answer"].count("<<")} for it in ds_math]
num, txt = topk(math_items, 1500, lambda x: x["s"])   # 2 数值代数(多步计算) vs 3 文字推导
for x in num:
    myriad_data.append({"core_id": 1, "core_name": "Math", "cluster_id": 2, "prompt": x["prompt"], "response": x["response"]})
for x in txt:
    myriad_data.append({"core_id": 1, "core_name": "Math", "cluster_id": 3, "prompt": x["prompt"], "response": x["response"]})

# ----------------------------------------------------------------------
# Core 2: 自然科学 (3000)
# ----------------------------------------------------------------------
print("    - [3/8] Core 2 自然科学...")
ds_sciq = load_dataset("allenai/sciq", split="train[:3000]")
sci_items = []
for item in ds_sciq:
    p = f"Question: {item['question']}\nContext: {item['support']}"
    sci_items.append({"prompt": p, "response": item["correct_answer"],
                      "s": score(item["question"] + "\n" + item["support"], EXP_PAT)})
exp, bas = topk(sci_items, 1500, lambda x: x["s"])   # 5 实验科学 vs 4 基础物化
for x in bas:
    myriad_data.append({"core_id": 2, "core_name": "Science", "cluster_id": 4, "prompt": x["prompt"], "response": x["response"]})
for x in exp:
    myriad_data.append({"core_id": 2, "core_name": "Science", "cluster_id": 5, "prompt": x["prompt"], "response": x["response"]})

# ----------------------------------------------------------------------
# Core 3/4: no_robots 按 category 语义分流 (各3000, Coding剔除)
# ----------------------------------------------------------------------
print("    - [4/8][5/8] Core 3 文学创意 + Core 4 商务对话 (按category)...")
ds_nr = load_dataset("HuggingFaceH4/no_robots", split="train")
buckets = {}
for item in ds_nr:
    msgs = item["messages"]
    if len(msgs) < 2:
        continue
    if not msgs[1]["content"].strip():
        continue
    buckets.setdefault(item.get("category", "?"), []).append(
        {"prompt": msgs[0]["content"], "response": msgs[1]["content"]})
print(f"      可用 bucket: { {k: len(v) for k, v in sorted(buckets.items())} }")

gen = buckets.get("Generation", [])
brs = buckets.get("Brainstorm", [])
gen_by_len = sorted(gen, key=lambda x: len(x["response"]), reverse=True)
c6 = take(gen_by_len, 1500, "Generation/长篇叙事")                       # 6 长篇叙事(最长的1500)
used = set(map(id, c6))
rest_gen_asc = [x for x in sorted(gen, key=lambda x: len(x["response"])) if id(x) not in used]
c7_pool = brs + rest_gen_asc
c7 = take(c7_pool, 1500, "Brainstorm+短Generation/诗歌意象")              # 7 诗歌意象
for x in c6:
    myriad_data.append({"core_id": 3, "core_name": "Creative_Arts", "cluster_id": 6, "prompt": x["prompt"], "response": x["response"]})
for x in c7:
    myriad_data.append({"core_id": 3, "core_name": "Creative_Arts", "cluster_id": 7, "prompt": x["prompt"], "response": x["response"]})

qa_pool = buckets.get("Open QA", []) + buckets.get("Closed QA", []) + buckets.get("Chat", [])
sum_pool = buckets.get("Summarize", []) + buckets.get("Rewrite", []) + buckets.get("Classify", []) + buckets.get("Extract", [])
c8 = take(sorted(qa_pool, key=lambda x: len(x["response"]), reverse=True), 1500, "问答/专业问答")
c9 = sum_pool[:1500]
if len(c9) < 1500:  # 万一不够, 用最短的Chat补齐并如实报告
    have = set(map(id, c8)) | set(map(id, c9))
    fill = [x for x in sorted(buckets.get("Chat", []), key=lambda x: len(x["response"])) if id(x) not in have]
    need = 1500 - len(c9)
    c9 = c9 + take(fill, need, "Chat补齐/公文摘要")
    print(f"      [!] 摘要类不够, 用{need}条短Chat补齐到1500")
for x in c8:
    myriad_data.append({"core_id": 4, "core_name": "Business_Dialogue", "cluster_id": 8, "prompt": x["prompt"], "response": x["response"]})
for x in c9:
    myriad_data.append({"core_id": 4, "core_name": "Business_Dialogue", "cluster_id": 9, "prompt": x["prompt"], "response": x["response"]})

# ----------------------------------------------------------------------
# Core 5/6/7: alpaca 9000 按语义打分定额三分
# ----------------------------------------------------------------------
print("    - [6/8][7/8][8/8] Core 5/6/7 alpaca语义三分...")
ds_al = load_dataset("tatsu-lab/alpaca", split="train[:9100]")  # 多取100条, 兜底空response过滤
al_items = []
for item in ds_al:
    p = item["instruction"] + (f"\n{item['input']}" if item.get("input") else "")
    if not item["output"].strip():
        continue
    f = p + "\n" + item["output"]
    al_items.append({"prompt": p, "response": item["output"],
                     "sc": score(f, C_PAT), "sl": score(f, L_PAT)})
    if len(al_items) >= 9000:
        break
assert len(al_items) == 9000, f"alpaca有效不足9000: {len(al_items)}"
print(f"      alpaca有效 {len(al_items)} 条 (constraint命中{sum(1 for x in al_items if x['sc']>0)}, "
      f"logic命中{sum(1 for x in al_items if x['sl']>0 and x['sc']==0)})")
cons, rest1 = topk(al_items, 3000, lambda x: x["sc"])   # 6 严格约束(最像约束的3000)
logi, rest2 = topk(rest1, 3000, lambda x: x["sl"])      # 5 逻辑思辨(剩余中最像逻辑的3000)
slot = take(rest2, 3000, "插槽")                          # 7 专属插槽(剩余)

fmt_first, neg_rest = topk(cons, 1500, lambda x: score(x["prompt"], FMT_PAT) - score(x["prompt"], NEG_PAT))
for x in fmt_first:  # 12 格式规范
    myriad_data.append({"core_id": 6, "core_name": "Constraint_Rules", "cluster_id": 12, "prompt": x["prompt"], "response": x["response"]})
for x in neg_rest:   # 13 负向安全(偏否定约束)
    myriad_data.append({"core_id": 6, "core_name": "Constraint_Rules", "cluster_id": 13, "prompt": x["prompt"], "response": x["response"]})

cause, philo = topk(logi, 1500, lambda x: score(x["prompt"], CAUSE_PAT))
for x in cause:  # 10 因果推演
    myriad_data.append({"core_id": 5, "core_name": "Logic_Philosophy", "cluster_id": 10, "prompt": x["prompt"], "response": x["response"]})
for x in philo:  # 11 哲学批判
    myriad_data.append({"core_id": 5, "core_name": "Logic_Philosophy", "cluster_id": 11, "prompt": x["prompt"], "response": x["response"]})

slot_by_len = sorted(slot, key=lambda x: len(x["response"]), reverse=True)
for x in slot_by_len[:1500]:  # 14 插槽A
    myriad_data.append({"core_id": 7, "core_name": "Custom_Slots", "cluster_id": 14, "prompt": x["prompt"], "response": x["response"]})
for x in slot_by_len[1500:3000]:  # 15 插槽B
    myriad_data.append({"core_id": 7, "core_name": "Custom_Slots", "cluster_id": 15, "prompt": x["prompt"], "response": x["response"]})

# ----------------------------------------------------------------------
# 断言 + 报告
# ----------------------------------------------------------------------
cc = Counter((r["core_id"], r["cluster_id"]) for r in myriad_data)
assert len(myriad_data) == 24000, f"总数 {len(myriad_data)} != 24000"
assert sorted(Counter(r["core_id"] for r in myriad_data).values()) == [3000] * 8, "core不平衡"
assert sorted(Counter(r["cluster_id"] for r in myriad_data).values()) == [1500] * 16, "cluster不平衡"
print(f"\n[✔] 组装完毕: {len(myriad_data)}条, 8 core各3000, 16 cluster各1500")
print("    每宗门示例:")
for cid in range(16):
    r = next(x for x in myriad_data if x["cluster_id"] == cid)
    print(f"    宗门{cid}: core{r['core_id']} P:{r['prompt'][:60].replace(chr(10),' ')}...")

jsonl_path = "myriad_train_data_v2.jsonl"
print(f"[*] 写入 {jsonl_path}...")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for entry in myriad_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print("[✔] JSONL 完成")

# ----------------------------------------------------------------------
# 离线 Tokenize (与训练脚本兼容, schema不变)
# ----------------------------------------------------------------------
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
    tokenized_records.append({
        "input_ids": input_ids, "labels": labels,
        "core_id": item["core_id"], "cluster_id": item["cluster_id"],
        "length": len(full_ids),
    })

tokenized_records.sort(key=lambda x: x["length"])
torch.save(tokenized_records, "myriad_tokenized_cache_v2.pt")
print(f"[✔] 缓存完成: {len(tokenized_records)}条, "
      f"cluster覆盖{min(r['cluster_id'] for r in tokenized_records)}~{max(r['cluster_id'] for r in tokenized_records)}")
print("=" * 80)
