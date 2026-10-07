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
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from transformers import TextIteratorStreamer
import importlib.util

STOP_STRINGS = ["<end_of_turn>"]  # 训练语料回合分隔符是7个文本piece, 不是单token, 只能字符串级截停
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
DEVICE = "cuda:0"
_MODEL_LOCK = threading.Lock()  # 生成/探测/热插拔互斥, 换权重时不撕裂正在跑的前向


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOK, WRAPPERS, B_NORMS
    MODEL, TOK, WRAPPERS = probemod.load_system()
    with torch.no_grad():
        B_NORMS = [{
            "layer": 18 + i,
            "shared": w.shared_lora_B.weight.detach().float().norm().item(),
            "macro": w.macro_lora_B.detach().float().norm().item(),
            "micro": w.micro_lora_B.detach().float().norm().item(),
        } for i, w in enumerate(WRAPPERS)]
    print("[server] 就绪: 12层MoE + 探测器", flush=True)
    yield


app = FastAPI(title="Myriad-MoE Chat", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_logger(req: Request, exc: RequestValidationError):
    try:
        body = (await req.body())[:500].decode("utf-8", "replace")
    except Exception:
        body = "<unreadable>"
    print(f"[!] 422 {req.url.path} body={body} errors={exc.errors()}", flush=True)
    return JSONResponse({"error": "请求体校验失败", "detail": exc.errors()}, status_code=422)


class ChatReq(BaseModel):
    prompt: str = ""
    max_tokens: int = 128

    @field_validator("prompt", mode="before")
    @classmethod
    def _s(cls, v):
        return v if isinstance(v, str) else ("" if v is None else str(v))

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _i(cls, v):
        try:
            return int(v)
        except Exception:
            return 128


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
    global B_NORMS
    import time
    if not req.weight or not os.path.isfile(req.weight):
        return JSONResponse({"error": f"权重文件不存在: {req.weight}"}, status_code=404)
    fp = req.weight if os.path.isabs(req.weight) else os.path.join(BASE, req.weight)
    try:
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
        return {"ok": True, "weight": req.weight, "secs": round(time.time() - t0, 1),
                "layers": len(WRAPPERS), "b_norms": B_NORMS}
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


def fmt(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


@app.get("/api/health")
def health():
    import torch as _t
    return {"ok": MODEL is not None,
            "vram_gb": round(_t.cuda.memory_allocated() / 1024**3, 2),
            "layers": [f"layer_{18+i}" for i in range(12)] if MODEL else []}


@app.post("/api/chat")
def chat(req: ChatReq):
    if not req.prompt.strip():
        return JSONResponse({"error": "prompt为空"}, status_code=400)
    try:
        ids = TOK.encode(fmt(req.prompt), return_tensors="pt").to(DEVICE)
        streamer = TextIteratorStreamer(TOK, skip_prompt=True, skip_special_tokens=False)

        def run():
            with _MODEL_LOCK, torch.no_grad():
                MODEL.generate(ids, max_new_tokens=max(1, req.max_tokens), do_sample=False,
                               repetition_penalty=1.1,
                               stopping_criteria=stop_criteria(),
                               pad_token_id=TOK.pad_token_id or 0,
                               streamer=streamer, use_cache=True)

        threading.Thread(target=run, daemon=True).start()

        def events():
            buf = []
            for tok in streamer:
                buf.append(tok)
                yield f"data: {json.dumps({'token': tok}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True, 'text': re.split(r'<end_of_turn>?', ''.join(buf))[0]})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")
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
