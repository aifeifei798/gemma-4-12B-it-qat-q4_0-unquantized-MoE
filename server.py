#!/usr/bin/env python3
# ==============================================================================
# server.py -- Myriad-MoE v2.3 FastAPI 服务端 (chat SSE流 + 路由探测)
#   POST /api/chat  {prompt, max_tokens}   -> SSE token流
#   POST /api/probe {prompt, response, domain} -> 路由诊断JSON (只看response段)
#   GET  /api/health
#   其余路径 -> web/dist 静态前端
# ==============================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import json
import re
import threading
import torch
import uuid
from contextlib import asynccontextmanager, contextmanager
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from transformers import TextIteratorStreamer
from transformers.generation import StoppingCriteria
import importlib.util

STOP_STRINGS = ["<end_of_turn>", "<turn|>"]  # 后者是多模态官方模板的回合关闭符
_CLEAN_RE = r"<end_of_turn>|<turn\|>"
_STOP_CRIT = None


def stop_criteria():
    """StopStringCriteria单例 (需TOK就绪后首次调用构建)."""
    global _STOP_CRIT
    if _STOP_CRIT is None:
        from transformers.generation import StoppingCriteriaList, StopStringCriteria
        _STOP_CRIT = StoppingCriteriaList([StopStringCriteria(tokenizer=TOK, stop_strings=STOP_STRINGS)])
    return _STOP_CRIT

BASE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("probe", os.path.join(BASE, "3.infer_probe.py"))
probemod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probemod)
trainmod = probemod.trainmod  # 3.infer_probe 内部已加载训练模块, 复用 (load_weights_dict)

MODEL, TOK, WRAPPERS, B_NORMS = None, None, None, None
PROC = None  # 多模态处理器 (Gemma4UnifiedProcessor, 有附件时用官方模板组装输入)
DEVICE = "cuda:0"
_MODEL_LOCK = threading.Lock()  # 生成/探测/热插拔互斥, 换权重时不撕裂正在跑的前向
_DATA_LOCK = threading.Lock()  # 错题本文件追加锁

# ---- 附件上传 ----
UPLOAD_DIR = "/tmp/opencode/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUD_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus"}
VID_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
MAX_UPLOAD_MB = 50
# 遥测透传白名单: 只带与prompt对齐的多模态张量 (mask类与response拼接错位, 不带)
MM_TELEMETRY_KEYS = ("pixel_values", "image_position_ids",
                     "input_features", "input_features_mask")

# ---- 实验驾驶舱: 全局操控状态 (内存单例, 改完即对后续请求生效) ----
STEER = {
    "routing_temperature": 1.0,
    "disabled_macros": [],
    "disabled_clusters": [],
    "disable_shared": False,
    "disable_macro": False,
    "disable_micro": False,
    "disable_all_moe": False,
    "max_context_tokens": 4096,
}
BAD_CASES = os.path.join(BASE, "hard_cases_v4.jsonl")

# 单次请求前缀手操 (前端剥掉前缀再填字段; 服务端同样识别, 直连curl可用)
PREFIXES = {
    "/force-code": {"force_macro": 0},
    "/no-micro": {"disable_micro": True},
    "/no-macro": {"disable_macro": True},
    "/base-only": {"disable_all_moe": True},
    "/cold": {"routing_temperature": 0.1},
    "/wild": {"routing_temperature": 2.5},
}


def parse_prefix(prompt: str):
    s = prompt.lstrip()
    for k, v in PREFIXES.items():
        if s.startswith(k) and (len(s) == len(k) or s[len(k)] in (" ", "\n", "\t")):
            return s[len(k):].lstrip(), dict(v)
    return prompt, {}


class CancelCriteria(StoppingCriteria):
    """中断按钮: 前端Abort -> SSE生成器关闭 -> finally置旗 -> 下个token停generate."""

    def __init__(self, flag: threading.Event):
        self.flag = flag

    def __call__(self, input_ids, scores, **kwargs):
        return bool(self.flag.is_set())


def _mget(m, k):
    if isinstance(m, dict):
        return m.get(k, "")
    return getattr(m, k, "") or ""


def build_prompt(history, prompt: str) -> str:
    """多轮历史拼成Gemma回合格式 (role assistant/model/ai->model, 其余->user)."""
    parts = []
    for m in history or []:
        role = _mget(m, "role") or "user"
        content = _mget(m, "content") or ""
        tag = "model" if role in ("assistant", "model", "ai") else "user"
        parts.append(f"<start_of_turn>{tag}\n{content}<end_of_turn>\n")
    parts.append(fmt(prompt))
    return "".join(parts)


def apply_steering(wrappers, st):
    """把操控状态灌进12层wrapper (T+掩码+分支开关). 全禁时门控内退化为不禁, 防NaN."""
    t = min(max(float(st.get("routing_temperature", 1.0) or 1.0), 0.05), 5.0)
    dm = sorted({int(x) for x in (st.get("disabled_macros") or []) if 0 <= int(x) < 8})
    dc = sorted({int(x) for x in (st.get("disabled_clusters") or []) if 0 <= int(x) < 16})
    for w in wrappers:
        w.steer_temp = t
        w.steer_disabled_macros = dm
        w.steer_disabled_clusters = dc
        w.steer_disable_shared = bool(st.get("disable_shared", False))
        w.steer_disable_macro = bool(st.get("disable_macro", False))
        w.steer_disable_micro = bool(st.get("disable_micro", False))
        w.steer_disable_all = bool(st.get("disable_all_moe", False))


@contextmanager
def _steer_override(over):
    """按次覆盖: 全局STEER打底 + over补丁, 退出恢复. 调用方须已持 _MODEL_LOCK."""
    if not over:
        yield dict(STEER)
        return
    saved = [(w, w.steer_temp, list(w.steer_disabled_macros), list(w.steer_disabled_clusters),
              w.steer_disable_shared, w.steer_disable_macro, w.steer_disable_micro,
              w.steer_disable_all) for w in WRAPPERS]
    merged = dict(STEER)
    for k, v in over.items():
        if v is not None and k in merged:
            merged[k] = v
    if over.get("force_macro") is not None:
        try:
            fm = int(over["force_macro"])
            if 0 <= fm < 8:
                merged["disabled_macros"] = [i for i in range(8) if i != fm]
        except Exception:
            pass
    try:
        apply_steering(WRAPPERS, merged)
        yield merged
    finally:
        for w, t, dm, dc, ds, dma, dmi, dall in saved:
            w.steer_temp = t
            w.steer_disabled_macros = dm
            w.steer_disabled_clusters = dc
            w.steer_disable_shared = ds
            w.steer_disable_macro = dma
            w.steer_disable_micro = dmi
            w.steer_disable_all = dall


def route_summary(prompt_ids, response_text, st=None, mm_extra=None):
    """response段路由聚合 + 终端心电图日志. 调用方须已持 _MODEL_LOCK. 一次prefill成本, 非零成本.
    mm_extra: 多模态透传 (仅白名单key, 与prompt对齐, 不含mask)."""
    import collections
    st = st or STEER
    seg_start = prompt_ids.shape[1]
    if response_text and response_text.strip():
        resp_ids = TOK.encode(response_text, add_special_tokens=False,
                              return_tensors="pt").to(DEVICE)
        full = torch.cat([prompt_ids, resp_ids], dim=1)
    else:
        full, seg_start = prompt_ids, 0
    diags = probemod.probe_forward(MODEL, WRAPPERS, full, **(mm_extra or {}))
    mh, ch = collections.Counter(), collections.Counter()
    e_s = e_m = e_u = e_b = 0.0
    n_tok = 0
    for d in diags:
        mh.update(d["macro_pred"][seg_start:].tolist())
        ch.update(d["cluster_pred"][seg_start:].tolist())
        e_s += d["n_shared"][seg_start:].sum().item()
        e_m += d["n_macro"][seg_start:].sum().item()
        e_u += d["n_micro"][seg_start:].sum().item()
        e_b += d["n_base"][seg_start:].sum().item()
        n_tok += max(len(d["macro_pred"]) - seg_start, 1)
    n_tok = max(n_tok, 1)
    e_b = max(e_b, 1e-9)
    top2 = mh.most_common(2)
    m1, m1n = top2[0] if top2 else (-1, 0)
    m2, m2n = top2[1] if len(top2) > 1 else (-1, 0)
    c1, c1n = ch.most_common(1)[0] if ch else (-1, 0)
    CN, CL = probemod.CORE_NAMES, probemod.CLUSTER_NAMES
    masks = (st.get("disabled_macros") or []) + (st.get("disabled_clusters") or [])
    print(f"[路由 L18-29] 天王:{CN[m1] if m1 >= 0 else '-'} {m1n / n_tok:.0%}"
          + (f" + {CN[m2]} {m2n / n_tok:.0%}" if m2 >= 0 else "")
          + f" | 宗门:{CL[c1] if c1 >= 0 else '-'} {c1n / n_tok:.0%}"
          + f" | T={st.get('routing_temperature', 1.0)}"
          + (f" masks={masks}" if masks else "")
          + (f" | 微关" if st.get("disable_micro") else "")
          + (f" | 纯基座" if st.get("disable_all_moe") else "")
          + f" | {n_tok // 12}tok", flush=True)
    return {
        "macro_top": [m1, m2],
        "macro_names": [CN[m1] if m1 >= 0 else "", CN[m2] if m2 >= 0 else ""],
        "macro_conf": round(m1n / n_tok, 3) if m1 >= 0 else 0.0,
        "cluster_top": c1,
        "cluster_name": CL[c1] if c1 >= 0 else "",
        "macro_hist": [mh.get(i, 0) / n_tok for i in range(8)],
        "cluster_hist": [ch.get(i, 0) / n_tok for i in range(16)],
        "energy": {"shared": e_s / e_b, "macro": e_m / e_b, "micro": e_u / e_b},
    }


def build_mm_inputs(history, prompt_text, file_ids):
    """多模态输入组装 (官方模板, 非训练fmt). 有附件时历史仅支持纯文本轮.
    返回 generate/probe 直用的 kwargs (tensors已上DEVICE)."""
    if PROC is None:
        raise ValueError("多模态处理器未就绪 (纯文本模式)")
    images, audios = [], []
    for fid in file_ids or []:
        p = os.path.join(UPLOAD_DIR, os.path.basename(str(fid)))
        if not os.path.isfile(p):
            raise ValueError(f"附件不存在: {fid}")
        ext = os.path.splitext(p)[1].lower()
        if ext in IMG_EXTS:
            from PIL import Image
            images.append(Image.open(p).convert("RGB"))
        elif ext in AUD_EXTS:
            audios.append(p)
        else:
            raise ValueError(f"暂不支持: {fid} (视频二期)")
    if not images and not audios:
        raise ValueError("files为空或无效")
    content = ([{"type": "image"} for _ in images]
               + [{"type": "audio"} for _ in audios]
               + [{"type": "text", "text": prompt_text}])
    messages = []
    for m in history or []:
        role = _mget(m, "role") or "user"
        messages.append({"role": "assistant" if role in ("assistant", "model", "ai") else "user",
                         "content": _mget(m, "content") or ""})
    messages.append({"role": "user", "content": content})
    text = PROC.apply_chat_template(messages, add_generation_prompt=True)
    kw = {"text": text}
    if images:
        kw["images"] = images[0] if len(images) == 1 else images
    if audios:
        kw["audio"] = audios[0] if len(audios) == 1 else audios
    inputs = PROC(**kw, return_tensors="pt")
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(DEVICE)
            # 视觉/音频塔是bf16 (与文本路同), 处理器吐float32, 不转就炸LayerNorm;
            # mask类不动 (bool/long或对齐用float)
            if v.is_floating_point() and "mask" not in k:
                v = v.to(torch.bfloat16)
        out[k] = v
    return out


def _do_hotswap(weight):
    """热插拔实装 (/api/hotswap 与 /admin/reload-weights 共用)."""
    import time
    global B_NORMS
    if not weight or not os.path.isfile(weight):
        return None, f"权重文件不存在: {weight}"
    fp = weight if os.path.isabs(weight) else os.path.join(BASE, weight)
    t0 = time.time()
    with _MODEL_LOCK:
        sd = torch.load(fp, map_location="cpu", weights_only=False)
        trainmod.load_weights_dict(WRAPPERS, 18, sd)
        with torch.no_grad():
            B_NORMS = [{
                "layer": 18 + i,
                "shared": w.shared_lora_B.weight.detach().float().norm().item(),
                "macro": w.macro_lora_B.detach().float().norm().item(),
                "micro": w.micro_lora_B.detach().float().norm().item(),
            } for i, w in enumerate(WRAPPERS)]
    return {"ok": True, "weight": weight, "secs": round(time.time() - t0, 1),
            "layers": len(WRAPPERS), "b_norms": B_NORMS}, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOK, WRAPPERS, B_NORMS, PROC
    MODEL, TOK, WRAPPERS, PROC = probemod.load_system()
    with torch.no_grad():
        B_NORMS = [{
            "layer": 18 + i,
            "shared": w.shared_lora_B.weight.detach().float().norm().item(),
            "macro": w.macro_lora_B.detach().float().norm().item(),
            "micro": w.micro_lora_B.detach().float().norm().item(),
        } for i, w in enumerate(WRAPPERS)]
    apply_steering(WRAPPERS, STEER)  # 全局操控初值灌入12层
    print("[server] 就绪: 12层MoE + 探测器 + 驾驶舱", flush=True)
    yield


app = FastAPI(title="Myriad-MoE Chat", lifespan=lifespan)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """附件上传 (图/音; 视频二期). 返回服务端本地id, 聊天时files字段引用."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext in IMG_EXTS:
        kind = "image"
    elif ext in AUD_EXTS:
        kind = "audio"
    elif ext in VID_EXTS:
        return JSONResponse({"error": "视频二期, 先传图片/音频"}, status_code=400)
    else:
        return JSONResponse({"error": f"不支持的格式: {ext or '?'} (仅图/音)"}, status_code=400)
    try:
        data = await file.read()
    except Exception as e:
        return JSONResponse({"error": f"读取失败: {e}"}, status_code=400)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse({"error": f"超限{MAX_UPLOAD_MB}MB"}, status_code=413)
    if len(data) == 0:
        return JSONResponse({"error": "空文件"}, status_code=400)
    fid = uuid.uuid4().hex + ext
    with open(os.path.join(UPLOAD_DIR, fid), "wb") as f:
        f.write(data)
    return {"ok": True, "id": fid, "kind": kind,
            "name": file.filename, "bytes": len(data)}


@app.exception_handler(RequestValidationError)
async def validation_logger(req: Request, exc: RequestValidationError):
    try:
        body = (await req.body())[:500].decode("utf-8", "replace")
    except Exception:
        body = "<unreadable>"
    print(f"[!] 422 {req.url.path} body={body} errors={exc.errors()}", flush=True)
    return JSONResponse({"error": "请求体校验失败", "detail": exc.errors()}, status_code=422)


def _int_list(v):
    if v is None:
        return None
    if isinstance(v, (int, str)):
        v = [v]
    out = []
    for x in (v or []):
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


class ChatReq(BaseModel):
    prompt: str = ""
    max_tokens: int = 128
    history: list | None = None  # [{role, content}], 多轮上下文 (超出max_context_tokens左截断)
    files: list | None = None  # 附件id (/api/upload返回), 有附件走多模态官方模板
    # 按次操控覆盖 (None=沿用全局; 显式值优先于文本前缀)
    routing_temperature: float | None = None
    disabled_macros: list | None = None
    disabled_clusters: list | None = None
    disable_shared: bool | None = None
    disable_macro: bool | None = None
    disable_micro: bool | None = None
    disable_all_moe: bool | None = None
    force_macro: int | None = None

    @field_validator("prompt", mode="before")
    @classmethod
    def _s(cls, v):
        return v if isinstance(v, str) else ("" if v is None else str(v))

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _i(cls, v):
        try:
            return max(1, min(int(v), 1024))
        except Exception:
            return 128

    @field_validator("disabled_macros", "disabled_clusters", mode="before")
    @classmethod
    def _li(cls, v):
        return _int_list(v)


class ProbeReq(BaseModel):
    prompt: str = ""
    response: str = ""
    domain: int = -1

    @field_validator("prompt", "response", mode="before")
    @classmethod
    def _s(cls, v):
        return v if isinstance(v, str) else ("" if v is None else str(v))

    @field_validator("domain", mode="before")
    @classmethod
    def _i(cls, v):
        try:
            return int(v)
        except Exception:
            return -1


class HotswapReq(BaseModel):
    weight: str = "myriad_moe_patch_poetry.pt"

    @field_validator("weight", mode="before")
    @classmethod
    def _s(cls, v):
        return str(v) if v is not None else ""


@app.post("/api/hotswap")
def hotswap(req: HotswapReq):
    """热插拔权重: 不重启服务, 直接把新权重灌进正在跑的模型. 返回各层B范数供确认."""
    if MODEL is None:
        return JSONResponse({"error": "模型未就绪"}, status_code=503)
    try:
        out, err = _do_hotswap(req.weight)
        if err:
            return JSONResponse({"error": err}, status_code=404)
        return out
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


class ReloadReq(BaseModel):
    weight: str = "myriad_moe_hierarchical_weights_v3.pt"

    @field_validator("weight", mode="before")
    @classmethod
    def _s(cls, v):
        return str(v) if v is not None else ""


@app.post("/admin/reload-weights")
def reload_weights(req: ReloadReq):
    """/api/hotswap 别名 (驾驶舱命名). 基座常驻, 秒级原地覆写."""
    if MODEL is None:
        return JSONResponse({"error": "模型未就绪"}, status_code=503)
    try:
        out, err = _do_hotswap(req.weight)
        if err:
            return JSONResponse({"error": err}, status_code=404)
        return out
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/admin/gpu-clear")
def gpu_clear():
    """显存清道夫: gc + empty_cache, 回到待机水位."""
    import torch as _t
    ok = _t.cuda.is_available()
    before = _t.cuda.memory_allocated() / 1024**3 if ok else 0.0
    gc.collect()
    if ok:
        _t.cuda.empty_cache()
    after = _t.cuda.memory_allocated() / 1024**3 if ok else 0.0
    return {"ok": True, "before_gb": round(before, 2), "after_gb": round(after, 2)}


def fmt(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


@app.get("/api/health")
def health():
    import torch as _t
    ok = _t.cuda.is_available()
    return {"ok": MODEL is not None,
            "vram_gb": round(_t.cuda.memory_allocated() / 1024**3, 2) if ok else 0.0,
            "vram_reserved_gb": round(_t.cuda.memory_reserved() / 1024**3, 2) if ok else 0.0,
            "vram_peak_gb": round(_t.cuda.max_memory_allocated() / 1024**3, 2) if ok else 0.0,
            "layers": [f"layer_{18+i}" for i in range(12)] if MODEL else [],
            "steering": dict(STEER)}


@app.get("/api/domains")
def domains():
    """语义别名下发 (真源 domains.yaml, 经 3.infer_probe 加载)."""
    CN, CE = probemod.CORE_NAMES, probemod.CORE_EN
    CL, CLE = probemod.CLUSTER_NAMES, probemod.CLUSTER_EN
    CP = probemod.CLUSTER_PARENT
    cores = []
    for i in range(8):
        zh = CN[i][:-len(CE[i])] if CE[i] and CN[i].endswith(CE[i]) else CN[i]
        cores.append({"id": i, "zh": zh, "en": CE[i], "name": CN[i]})
    return {"cores": cores,
            "clusters": [{"id": i, "zh": CL[i], "en": CLE[i], "parent": CP[i]}
                         for i in range(16)]}


class SteeringReq(BaseModel):
    routing_temperature: float | None = None
    disabled_macros: list | None = None
    disabled_clusters: list | None = None
    disable_shared: bool | None = None
    disable_macro: bool | None = None
    disable_micro: bool | None = None
    disable_all_moe: bool | None = None
    max_context_tokens: int | None = None

    @field_validator("disabled_macros", "disabled_clusters", mode="before")
    @classmethod
    def _li(cls, v):
        return _int_list(v)


@app.get("/admin/steering")
def steering_get():
    return dict(STEER)


@app.post("/admin/steering")
def steering_set(req: SteeringReq):
    for k, v in req.model_dump().items():
        if v is None:
            continue
        if k == "routing_temperature":
            STEER[k] = min(max(float(v), 0.05), 5.0)
        elif k == "disabled_macros":
            STEER[k] = sorted({int(x) for x in v if 0 <= int(x) < 8})
        elif k == "disabled_clusters":
            STEER[k] = sorted({int(x) for x in v if 0 <= int(x) < 16})
        elif k == "max_context_tokens":
            STEER[k] = min(max(int(v), 256), 16384)
        else:
            STEER[k] = bool(v)
    if MODEL is not None:
        with _MODEL_LOCK:
            apply_steering(WRAPPERS, STEER)
    print(f"[操控] {STEER}", flush=True)
    return dict(STEER)


@app.post("/api/chat")
def chat(req: ChatReq):
    if MODEL is None:
        return JSONResponse({"error": "模型未就绪"}, status_code=503)
    if not req.prompt.strip():
        return JSONResponse({"error": "prompt为空"}, status_code=400)
    try:
        clean, pre = parse_prefix(req.prompt)
        over = dict(pre)
        for k in ("routing_temperature", "disabled_macros", "disabled_clusters",
                  "disable_shared", "disable_macro", "disable_micro",
                  "disable_all_moe", "force_macro"):
            v = getattr(req, k, None)
            if v is not None:
                over[k] = v
        file_ids = [str(x) for x in (req.files or [])]
        if file_ids:
            try:
                gen_inputs = build_mm_inputs(req.history, clean, file_ids)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            ids = gen_inputs["input_ids"]
            mm_extra = {k: v for k, v in gen_inputs.items()
                        if k in MM_TELEMETRY_KEYS}
            print(f"[多模态] 图/音输入就绪: ids={tuple(ids.shape)} "
                  f"keys={[k for k in gen_inputs if k != 'input_ids']}", flush=True)
        else:
            ids = TOK.encode(build_prompt(req.history, clean),
                             return_tensors="pt").to(DEVICE)
            gen_inputs = {"input_ids": ids}
            mm_extra = None
        mc = int(STEER.get("max_context_tokens", 4096) or 4096)
        if ids.shape[1] > mc:
            if file_ids:
                return JSONResponse(
                    {"error": f"多模态prompt超长({ids.shape[1]} tokens), 上限{mc}"},
                    status_code=413)
            ids = ids[:, -mc:]  # 纯文本超长左截断, 保尾部指令
            gen_inputs = {"input_ids": ids}
            print(f"[!] 上下文超长已左截断到{mc}", flush=True)
        max_new = max(1, min(int(req.max_tokens or 128), 1024))
        streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=False)
        cancel = threading.Event()
        gen_err = {}

        def run():
            try:
                with _MODEL_LOCK, _steer_override(over), torch.no_grad():
                    from transformers.generation import StoppingCriteriaList
                    crit = StoppingCriteriaList(
                        list(stop_criteria()) + [CancelCriteria(cancel)])
                    MODEL.generate(**gen_inputs, max_new_tokens=max_new, do_sample=False,
                                   repetition_penalty=1.1,
                                   stopping_criteria=crit,
                                   pad_token_id=TOK.pad_token_id or 0,
                                   streamer=streamer, use_cache=True)
            except Exception as e:
                gen_err["e"] = f"{type(e).__name__}: {e}"
                print(f"[!] 生成异常: {gen_err['e']}", flush=True)
                try:
                    streamer.end()  # 别让前端挂死: 结束流, done包里带error
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()

        def events():
            try:
                buf = []
                for tok in streamer:
                    buf.append(tok)
                    yield f"data: {json.dumps({'token': tok}, ensure_ascii=False)}\n\n"
                text = re.split(_CLEAN_RE, "".join(buf))[0]
                payload = {"done": True, "text": text,
                           "stopped": bool(cancel.is_set())}
                if not text and gen_err.get("e"):
                    payload["error"] = gen_err["e"]
                try:  # 生成后遥测: 一次prefill成本, 给终端心电图 + 消息徽章
                    with _MODEL_LOCK, _steer_override(over) as st:
                        payload["meta"] = route_summary(ids, text, st, mm_extra)
                except Exception as e:
                    print(f"[!] 遥测失败: {type(e).__name__}: {e}", flush=True)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            finally:
                cancel.set()  # 客户端断开(SSE关闭) -> 后台generate下个token停, 锁释放

        return StreamingResponse(events(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


# ---- OpenAI 兼容网关 (Cherry Studio / Continue / NextChat 直连) ----

class V1Msg(BaseModel):
    role: str = "user"
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _s(cls, v):
        if isinstance(v, str):
            return v
        if isinstance(v, list):  # 多模态content块: 只拼文本
            return "".join(p.get("text", "") for p in v
                            if isinstance(p, dict) and p.get("type") == "text")
        return "" if v is None else str(v)


class V1ChatReq(BaseModel):
    messages: list[V1Msg] = []
    max_tokens: int = 128
    stream: bool = True
    # 操控透传 (OpenAI客户端不发, 自家UI/脚本可带)
    routing_temperature: float | None = None
    disable_micro: bool | None = None
    disable_macro: bool | None = None
    disable_all_moe: bool | None = None
    force_macro: int | None = None

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _i(cls, v):
        try:
            return max(1, min(int(v), 1024))
        except Exception:
            return 128


def v1_to_prompt(messages) -> str:
    parts = []
    for m in messages or []:
        role = "model" if m.role == "assistant" else "user"
        parts.append(f"<start_of_turn>{role}\n{m.content}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _prep_ids(prompt_text):
    ids = TOK.encode(prompt_text, return_tensors="pt").to(DEVICE)
    mc = int(STEER.get("max_context_tokens", 4096) or 4096)
    if ids.shape[1] > mc:
        ids = ids[:, -mc:]
        print(f"[!] 上下文超长已左截断到{mc}", flush=True)
    return ids


@app.get("/v1/models")
def v1_models():
    return {"object": "list",
            "data": [{"id": "myriad-moe-v3", "object": "model",
                      "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def v1_chat(req: V1ChatReq):
    if MODEL is None:
        return JSONResponse({"error": "模型未就绪"}, status_code=503)
    if not req.messages:
        return JSONResponse({"error": "messages为空"}, status_code=400)
    try:
        ids = _prep_ids(v1_to_prompt(req.messages))
        max_new = max(1, int(req.max_tokens or 128))
        over = {k: getattr(req, k) for k in
                ("routing_temperature", "disable_micro", "disable_macro",
                 "disable_all_moe", "force_macro")
                if getattr(req, k) is not None}
        streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=False)
        cancel = threading.Event()
        gen_err = {}

        def run():
            try:
                with _MODEL_LOCK, _steer_override(over), torch.no_grad():
                    from transformers.generation import StoppingCriteriaList
                    crit = StoppingCriteriaList(
                        list(stop_criteria()) + [CancelCriteria(cancel)])
                    MODEL.generate(ids, max_new_tokens=max_new, do_sample=False,
                                   repetition_penalty=1.1,
                                   stopping_criteria=crit,
                                   pad_token_id=TOK.pad_token_id or 0,
                                   streamer=streamer, use_cache=True)
            except Exception as e:
                gen_err["e"] = f"{type(e).__name__}: {e}"
                print(f"[!] 生成异常: {gen_err['e']}", flush=True)
                try:
                    streamer.end()
                except Exception:
                    pass

        th = threading.Thread(target=run, daemon=True)
        th.start()
        cid = "chatcmpl-" + uuid.uuid4().hex[:8]

        if req.stream:
            def v1_events():
                try:
                    buf = []
                    for tok in streamer:
                        buf.append(tok)
                        yield ("data: " + json.dumps(
                            {"id": cid, "object": "chat.completion.chunk",
                             "choices": [{"index": 0, "delta": {"content": tok}}]},
                            ensure_ascii=False) + "\n\n")
                    text = re.split(_CLEAN_RE, "".join(buf))[0]
                    meta = None
                    try:
                        with _MODEL_LOCK, _steer_override(over) as st:
                            meta = route_summary(ids, text, st)
                    except Exception as e:
                        print(f"[!] 遥测失败: {type(e).__name__}: {e}", flush=True)
                    tail = {"id": cid, "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "stop"}]}
                    if meta:
                        tail["myriad_meta"] = meta
                    yield "data: " + json.dumps(tail, ensure_ascii=False) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    cancel.set()  # Cherry Studio点停止 -> 连接断开 -> 后台generate停
            return StreamingResponse(v1_events(), media_type="text/event-stream")

        buf = []
        for tok in streamer:
            buf.append(tok)
        th.join(timeout=900)
        text = re.split(_CLEAN_RE, "".join(buf))[0]
        if not text and gen_err.get("e"):
            return JSONResponse({"error": gen_err["e"]}, status_code=500)
        meta = None
        try:
            with _MODEL_LOCK, _steer_override(over) as st:
                meta = route_summary(ids, text, st)
        except Exception as e:
            print(f"[!] 遥测失败: {type(e).__name__}: {e}", flush=True)
        out = {"id": cid, "object": "chat.completion",
               "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}]}
        if meta:
            out["myriad_meta"] = meta
        return out
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


class IntentReq(BaseModel):
    prompt: str = ""

    @field_validator("prompt", mode="before")
    @classmethod
    def _s(cls, v):
        return v if isinstance(v, str) else ("" if v is None else str(v))


@app.post("/v1/analyze/intent")
def analyze_intent(req: IntentReq):
    """意图瞬时分类: 一次prefill前向的成本 (只编码不解码), 返回天王/宗门归属.
    注意 prompt段含模板token, 命中率仅供参考, 别当真值."""
    import time
    if MODEL is None:
        return JSONResponse({"error": "模型未就绪"}, status_code=503)
    if not req.prompt.strip():
        return JSONResponse({"error": "prompt为空"}, status_code=400)
    try:
        t0 = time.time()
        ids = _prep_ids(fmt(req.prompt))
        with _MODEL_LOCK, torch.no_grad():
            diags = probemod.probe_forward(MODEL, WRAPPERS, ids)
        import collections
        mh = collections.Counter(sum([d["macro_pred"].tolist() for d in diags], []))
        ch = collections.Counter(sum([d["cluster_pred"].tolist() for d in diags], []))
        tot = max(sum(mh.values()), 1)
        m1, m1n = mh.most_common(1)[0]
        c1, c1n = ch.most_common(1)[0]
        CN, CE = probemod.CORE_NAMES, probemod.CORE_EN
        CL, CLE = probemod.CLUSTER_NAMES, probemod.CLUSTER_EN
        return {"macro_id": m1, "macro_name": CN[m1], "macro_en": CE[m1],
                "macro_conf": round(m1n / tot, 3),
                "cluster_id": c1, "cluster_name": CL[c1], "cluster_en": CLE[c1],
                "cluster_conf": round(c1n / tot, 3),
                "ms": round((time.time() - t0) * 1000),
                "note": "prompt段聚合(含模板token), 仅供参考"}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


class BadcaseReq(BaseModel):
    prompt: str = ""
    response: str = ""
    pred_macro: int | None = None
    pred_cluster: int | None = None
    correct_macro: int = -1
    correct_cluster: int = -1
    note: str = ""

    @field_validator("prompt", "response", "note", mode="before")
    @classmethod
    def _s(cls, v):
        return v if isinstance(v, str) else ("" if v is None else str(v))


@app.get("/data/badcases")
def list_badcases(limit: int = 20):
    """错题本读取: 最近N条 (驾驶舱数据tab展示)."""
    try:
        with _DATA_LOCK:
            if not os.path.isfile(BAD_CASES):
                return {"ok": True, "total": 0, "cases": []}
            with open(BAD_CASES, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
        out = []
        for ln in lines[-max(1, min(limit, 200)):][::-1]:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return {"ok": True, "total": len(lines), "cases": out}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/data/mark-badcase")
def mark_badcase(req: BadcaseReq):
    """错题本飞轮: 路由失误打标, 追加 hard_cases_v4.jsonl, 供下轮增量训练."""
    import time
    if not req.prompt.strip():
        return JSONResponse({"error": "prompt为空"}, status_code=400)
    if not 0 <= req.correct_macro < 8:
        return JSONResponse({"error": "correct_macro须0-7"}, status_code=400)
    rec = {"ts": round(time.time(), 1), "prompt": req.prompt,
           "response": req.response, "pred_macro": req.pred_macro,
           "pred_cluster": req.pred_cluster, "correct_macro": req.correct_macro,
           "correct_cluster": req.correct_cluster,
           "steering": dict(STEER), "note": req.note}
    try:
        with _DATA_LOCK:
            with open(BAD_CASES, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(BAD_CASES, encoding="utf-8") as f:
                n = sum(1 for _ in f)
        return {"ok": True, "path": "hard_cases_v4.jsonl", "total": n}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/api/probe")
def probe(req: ProbeReq):
    import collections
    try:
        # response为空时退化为分析prompt段, 保证UI永远有图可画
        seg = "response" if req.response.strip() else "prompt"
        prompt_ids = TOK.encode(fmt(req.prompt), return_tensors="pt").to(DEVICE)
        if seg == "response":
            resp_ids = TOK.encode(req.response, add_special_tokens=False,
                                  return_tensors="pt").to(DEVICE)
            full = torch.cat([prompt_ids, resp_ids], dim=1)
            start = prompt_ids.shape[1]
        else:
            full, start = prompt_ids, 0
        with _MODEL_LOCK:
            diags = probemod.probe_forward(MODEL, WRAPPERS, full)
        mh, ch = collections.Counter(), collections.Counter()
        e_s = e_m = e_u = e_b = 0.0
        n_tok = 0
        for d in diags:
            mh.update(d["macro_pred"][start:].tolist())
            ch.update(d["cluster_pred"][start:].tolist())
            e_s += d["n_shared"][start:].sum().item()
            e_m += d["n_macro"][start:].sum().item()
            e_u += d["n_micro"][start:].sum().item()
            e_b += d["n_base"][start:].sum().item()
            n_tok += max(len(d["macro_pred"]) - start, 1)
        n_tok = max(n_tok, 1)
        macro_hist = [mh.get(i, 0) / n_tok for i in range(8)]
        cluster_hist = [ch.get(i, 0) / n_tok for i in range(16)]
        hit = (mh.get(req.domain, 0) / n_tok) if req.domain >= 0 else None
        # 中间层前12个所选段token明细
        d0 = diags[len(diags) // 2]
        detail = []
        rids = full[0, start:start + 12].tolist()
        for k, t in enumerate(rids):
            detail.append({"tok": TOK.decode([t]),
                           "macro": d0["macro_idx"][start + k].tolist(),
                           "macro_w": [round(v, 2) for v in d0["macro_w"][start + k].tolist()],
                           "cluster": d0["cluster_idx"][start + k].tolist()})
        e_b = max(e_b, 1e-9)
        return {
            "segment": seg,
            "b_norms": B_NORMS,
            "energy": {"shared": e_s / e_b, "macro": e_m / e_b, "micro": e_u / e_b},
            "macro_hist": macro_hist, "cluster_hist": cluster_hist,
            "domain_hit": hit,
            "core_names": probemod.CORE_NAMES,
            "cluster_names": probemod.CLUSTER_NAMES,
            "cluster_parent": probemod.CLUSTER_PARENT,
            "tokens": detail,
        }
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


# ---- 静态前端 (web/dist) ----
DIST = os.path.join(BASE, "web", "dist")
if os.path.isdir(DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        fp = os.path.join(DIST, path)
        if path and os.path.isfile(fp):
            return FileResponse(fp)
        return FileResponse(os.path.join(DIST, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
