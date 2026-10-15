"""LuxTTS Studio —— FastAPI 后端。

**本文件不写任何生成逻辑**，只做「收 HTTP → 调 `app/` 的模块级函数 → 存盘 → 回 JSON」。
生成核心在 `app/core.py`，音色库在 `app/voices.py`，记录在 `app/history.py`。

两个刻意的设计：

1. **启动时不加载模型。** 加载要 4~5 秒，而界面验证脚本需要反复起停服务。
   `/api/bootstrap` 只报「有没有加载」，第一次真正合成时才装载。
2. **合成走「提交 + 轮询」而不是同步接口。** 冷启动第一次合成要十几秒
   （模型装载 + CUDA 预热），同步请求会让界面卡住且没法显示进度。

端口只有一处来源：`app/config.py` 的 `HOST` / `PORT`。
"""

from __future__ import annotations

import logging
import shutil
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import config, core, history, voices  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("luxtts.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(__file__).resolve().parent / "_uploads"
UPLOAD_MAX_MB = 100

app = FastAPI(title="LuxTTS Studio", docs_url=None, redoc_url=None)

# --------------------------------------------------------------------------
# 任务表
# --------------------------------------------------------------------------
_JOBS: dict[str, dict] = {}
# **必须是 RLock**：`_JOBS_LOCK` 会在「取出job → 交给纯函数转 JSON」这种结构里被嵌套获取。
# 普通 Lock 不可重入，同线程第二次加锁就是永久自死锁 —— 表现是任务永远停在「生成中」，
# 而日志里什么异常都没有（不是崩溃，是死等）。别改回 Lock。
_JOBS_LOCK = threading.RLock()
# 生成串行化：并发跑会把 4 GiB 的显存打满，失败方式还很难看
_GEN_LOCK = threading.Lock()
_JOB_KEEP = 50


def _job_new(kind: str) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "state": "queued",       # queued | running | done | error
        "stage": "排队中",
        "progress": 0,
        "created": time.time(),
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job["id"]] = job
        if len(_JOBS) > _JOB_KEEP:
            for old in sorted(_JOBS.values(), key=lambda j: j["created"])[:-_JOB_KEEP]:
                _JOBS.pop(old["id"], None)
    return job


def _job_update(job: dict, **kw) -> None:
    with _JOBS_LOCK:
        job.update(kw)


def _err_text(e: Exception) -> str:
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


# --------------------------------------------------------------------------
# 上传
# --------------------------------------------------------------------------
def _save_upload(upload: UploadFile) -> Path:
    """把上传的音频落到 ``web/_uploads/``（临时区），返回路径。

    这是临时文件 —— 保存音色时会解码转存进 ``voices/``，之后就可以删了。
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".webm"}:
        suffix = ".wav"
    dst = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    written = 0
    limit = UPLOAD_MAX_MB * 1024 * 1024
    with dst.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"文件超过 {UPLOAD_MAX_MB} MB")
            f.write(chunk)
    if written == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "上传的文件是空的")
    return dst


def _probe_audio(path: Path) -> float:
    """确认能解码并返回时长（秒）。上游那套是「传进来再报错」，这里提前拦。"""
    import librosa

    try:
        dur = float(librosa.get_duration(path=str(path)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"这个文件解不出音频（{type(e).__name__}）：{e}") from e
    if dur <= 0:
        raise HTTPException(400, "音频时长为 0，无法作为参考音频")
    return round(dur, 2)


def _resolve_upload(token: str) -> Path:
    """把界面给的 token 还原成 ``_uploads`` 里的真实路径（挡路径穿越）。"""
    name = Path(token or "").name
    if not name or name != token or ".." in name:
        raise HTTPException(400, "无效的上传标识")
    p = UPLOAD_DIR / name
    if not p.exists():
        raise HTTPException(404, "上传的文件已失效，请重新选择音频")
    return p


def _media_path(kind: str, filename: str) -> Path:
    """``/media/{outputs|voices}/{名}`` 的路径校验。"""
    roots = {"outputs": config.OUTPUTS_DIR, "voices": config.VOICES_DIR}
    if kind not in roots:
        raise HTTPException(404, "未知的资源类型")
    name = Path(filename).name
    if not name or name != filename or ".." in name:
        raise HTTPException(400, "非法路径")
    root = roots[kind].resolve()
    p = (root / name).resolve()
    if root not in p.parents and p != root:
        raise HTTPException(400, "非法路径")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "文件不存在")
    return p


# --------------------------------------------------------------------------
# 启动元信息
# --------------------------------------------------------------------------
@app.get("/api/bootstrap")
def bootstrap() -> dict:
    prof = core.active_profile()
    return {
        "profile": prof,
        "defaults": prof["defaults"],
        "param_ranges": prof["param_ranges"],
        "sr": prof["sr"],
        "trim": prof["trim"],
        "model": {
            "loaded": core.is_loaded(),
            "load_seconds": core.device_info()["load_seconds"],
            "path": "cuda/onnx" if prof["device"] == "cpu" else "pytorch",
        },
        "asr": {
            "engine": config.ASR_ENGINE,
            "enabled": config.ASR_ENABLED,
            "reason": core.device_info()["asr_reason"],
        },
        # 这里只给**个数**，不是列表 —— 列表走 /api/voices。
        # 刻意不叫 "voices"：那个键在 /api/voices 里是列表，同名不同义最容易踩。
        "voice_count": len(voices.list_voices()),
        "history": history.stats(),
        "upload_max_mb": UPLOAD_MAX_MB,
    }


@app.get("/api/voices")
def api_voices() -> dict:
    return {"voices": voices.list_voices()}


@app.get("/api/voices/{name}")
def api_voice(name: str) -> dict:
    try:
        return voices.get_voice(name)
    except voices.VoiceError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/api/voices")
def api_voice_create(
    file: UploadFile = File(...),
    name: str = Form(""),
    ref_text: str = Form(""),
    ref_duration: float | None = Form(None),
    rms: float | None = Form(None),
) -> dict:
    """新建音色：上传 → 解码转存 → 建 prompt 缓存（**这一步就完成落盘**）。"""
    if not config.ASR_ENABLED and not ref_text.strip():
        raise HTTPException(400, "自动转写已关闭，请手工填写参考文本")
    src = _save_upload(file)
    try:
        _probe_audio(src)
        meta = voices.save_voice(
            name or Path(file.filename or "音色").stem,
            src,
            ref_text=ref_text,
            ref_duration=ref_duration,
            rms=rms,
        )
    except voices.VoiceError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:              # 拿不到参考文本等
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("保存音色失败")
        raise HTTPException(500, f"保存失败：{_err_text(e)}") from e
    finally:
        src.unlink(missing_ok=True)
    return {"voice": meta}


@app.post("/api/voices/update")
def api_voice_update(payload: dict) -> dict:
    try:
        meta = voices.update_voice(
            payload.get("name", ""),
            ref_text=payload.get("ref_text"),
            ref_duration=payload.get("ref_duration"),
            rms=payload.get("rms"),
            rebuild=bool(payload.get("rebuild", True)),
        )
    except voices.VoiceError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("更新音色失败")
        raise HTTPException(500, f"更新失败：{_err_text(e)}") from e
    return {"voice": meta}


@app.post("/api/voices/rename")
def api_voice_rename(payload: dict) -> dict:
    try:
        meta = voices.rename_voice(payload.get("name", ""), payload.get("new_name", ""))
    except voices.VoiceError as e:
        raise HTTPException(400, str(e)) from e
    return {"voice": meta}


@app.post("/api/voices/retranscribe")
def api_voice_retranscribe(payload: dict) -> dict:
    try:
        meta = voices.retranscribe(payload.get("name", ""), duration=payload.get("duration"))
    except voices.VoiceError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001
        log.exception("重新转写失败")
        raise HTTPException(500, f"转写失败：{_err_text(e)}") from e
    return {"voice": meta}


@app.post("/api/voices/delete")
def api_voice_delete(payload: dict) -> dict:
    return voices.delete_voices(list(payload.get("names") or []))


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...), duration: float | None = Form(None)) -> dict:
    """试转写（**不落盘、不建音色**）。返回文本 + 时长，供界面确认后再保存。"""
    if not config.ASR_ENABLED:
        raise HTTPException(400, "自动转写已关闭（LUXTTS_ASR=0）")
    src = _save_upload(file)
    try:
        dur = _probe_audio(src)
        from app import asr as asr_mod

        # 注意这里**不 load_model()**：SenseVoice 走 sherpa-onnx 的纯 CPU 路径，
        # 和 TTS 模型互不相干。转写不该强迫用户付 5 秒的模型装载 ——
        # 特别是「先转写看看内容」这种只想试一下的场景。
        transcriber = asr_mod.get_transcriber()
        if transcriber is None:
            raise HTTPException(400, f"转写引擎不可用：{asr_mod.unavailable_reason()}")

        import librosa

        use = duration if duration and duration > 0 else None
        wav, _ = librosa.load(str(src), sr=16000, duration=use)
        text = (transcriber({"array": wav, "sampling_rate": 16000}).get("text") or "").strip()
        return {
            "text": text,
            "duration": dur,
            "engine": config.ASR_ENGINE,
            "token": src.name,          # 后面保存音色就复用这个已上传的文件
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("试转写失败")
        raise HTTPException(500, f"转写失败：{_err_text(e)}") from e


# --------------------------------------------------------------------------
# 合成任务
# --------------------------------------------------------------------------
@app.post("/api/upload")
def api_upload(file: UploadFile = File(...)) -> dict:
    """单独上传参考音频（临时试听用），返回 token 供提交任务时引用。"""
    src = _save_upload(file)
    try:
        return {"token": src.name, "duration": _probe_audio(src), "filename": file.filename}
    except HTTPException:
        src.unlink(missing_ok=True)
        raise


@app.post("/api/jobs")
def api_job_create(payload: dict) -> dict:
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "请输入要合成的文本")

    voice = (payload.get("voice") or "").strip() or None
    token = (payload.get("token") or "").strip()
    if not voice and not token:
        raise HTTPException(400, "请选择音色，或上传一段参考音频")

    job = _job_new("synthesize")
    threading.Thread(
        target=_run_synthesize, args=(job, text, voice, token, payload), daemon=True
    ).start()
    # 起跑后工作线程可能立刻改状态，所以**在同一把锁内取快照** —— 否则可能读到
    # 「stage 已更新 / result 还没写」这种撕裂的中间态。
    with _JOBS_LOCK:
        return {"job": _public_job(job)}


def _public_job(job: dict) -> dict:
    """把内部 job 摊成对外 JSON。**纯转换，不加锁** —— 锁由调用方拿。"""
    return {
        "id": job["id"],
        "kind": job["kind"],
        "state": job["state"],
        "stage": job["stage"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
        "created": job["created"],
    }


@app.get("/api/jobs/{job_id}")
def api_job_get(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在（可能是服务重启过）")
        return _public_job(job)


def _run_synthesize(job: dict, text: str, voice: str | None, token: str, payload: dict) -> None:
    """跑一次合成：建/取 prompt → 生成 → 落盘 → 记历史。**成功失败都记。**"""
    params = core.merged_params(payload.get("params"))
    ref_text = (payload.get("ref_text") or "").strip()
    out_path: Path | None = None
    with _GEN_LOCK:
        try:
            _job_update(job, state="running", stage="装载模型（首次较慢）", progress=5)
            t0 = time.perf_counter()
            core.load_model()

            if voice:
                _job_update(job, stage=f"准备音色「{voice}」", progress=25)
                prompt = voices.read_prompt(voice)
                ref_seconds = prompt.get("ref_seconds")
                ref_text = prompt.get("ref_text", ref_text)
            else:
                _job_update(job, stage="编码参考音频" + ("（含转写）" if not ref_text else ""), progress=25)
                prompt = core.encode_ref(
                    _resolve_upload(token),
                    duration=params["ref_duration"],
                    rms=params["rms"],
                    ref_text=ref_text or None,
                )
                ref_seconds = prompt.get("ref_seconds")
                ref_text = prompt.get("ref_text", ref_text)

            _job_update(job, stage=f"生成中（{params['num_steps']} 步）", progress=70)
            out = core.synthesize(text, prompt, params)

            out_path = history.next_output_path(text, voice)
            core.save_wav(out_path, out["wav"], out["sr"])

            entry = history.add(
                ok=True, text=text, voice=voice, params=out["params"],
                output=out_path, sr=out["sr"], seconds=out["seconds"],
                elapsed=out["elapsed"], ref_text=ref_text,
                mode="voice" if voice else "adhoc",
            )
            _job_update(
                job, state="done", stage="完成", progress=100,
                result={
                    "output": out_path.name,
                    "url": f"/media/outputs/{out_path.name}",
                    "sr": out["sr"],
                    "seconds": out["seconds"],
                    "elapsed": out["elapsed"],
                    "total_elapsed": round(time.perf_counter() - t0, 2),
                    "ref_seconds": ref_seconds,
                    "ref_text": ref_text,
                    "params": out["params"],
                    "history_id": entry["id"],
                    "voice": voice,
                },
            )
        except Exception as e:  # noqa: BLE001
            log.exception("合成失败")
            err = _err_text(e)
            try:
                history.add(
                    ok=False, text=text, voice=voice, params=params, error=err,
                    ref_text=ref_text, mode="voice" if voice else "adhoc",
                )
            except Exception:  # noqa: BLE001 —— 记历史本身失败也不能盖掉真正的错误
                log.exception("写合成记录失败")
            _job_update(job, state="error", stage="失败", progress=100, error=err)


# --------------------------------------------------------------------------
# 合成记录
# --------------------------------------------------------------------------
@app.get("/api/history")
def api_history(limit: int = 100, offset: int = 0) -> dict:
    return history.list_items(limit=max(1, min(500, limit)), offset=max(0, offset))


@app.post("/api/history/delete")
def api_history_delete(payload: dict) -> dict:
    return history.delete(list(payload.get("ids") or []))


@app.post("/api/history/clear")
def api_history_clear() -> dict:
    return history.clear()


# --------------------------------------------------------------------------
# 静态资源
# --------------------------------------------------------------------------
@app.get("/media/{kind}/{filename}")
def api_media(kind: str, filename: str) -> FileResponse:
    return FileResponse(_media_path(kind, filename))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico")
def favicon() -> JSONResponse:
    return JSONResponse({}, status_code=204)


def main() -> int:
    import argparse

    import faulthandler

    import uvicorn

    # 生成跑在后台线程里的 CUDA 上，真出问题时是**静默退出**：没有 traceback，
    # 没有异常，进程直接没了，try/except 一个都拦不住。faulthandler 能把
    # 崩溃瞬间所有线程的栈打到 stderr —— 这是唯一能看清「死在哪」的线索。
    faulthandler.enable(all_threads=True)

    ap = argparse.ArgumentParser(description="LuxTTS Studio")
    ap.add_argument("--open", action="store_true",
                    help="服务起来后自动打开浏览器（由 run_web.bat 传）")
    args = ap.parse_args()

    url = f"http://{config.HOST}:{config.PORT}"
    print("=" * 62)
    print("  LuxTTS Studio")
    print(f"  界面  {url}")
    print(f"  模型  {config.CHECKPOINTS}")
    print(f"  音色  {config.VOICES_DIR}")
    print(f"  产物  {config.OUTPUTS_DIR}")
    print("  （模型不在启动时加载，第一次合成时才装载）")
    print("=" * 62, flush=True)

    # 浏览器由这边开，而不是写在 bat 里：端口的唯一真源是 config.PORT，
    # bat 里再抄一份迟早会对不上（而且 for /f 读 Python 输出一旦失败是静默的）。
    # 延后一点再开，是为了让 uvicorn 先把端口占住，否则打开就是一个连接被拒。
    if args.open:
        import threading
        import webbrowser

        def _open_later() -> None:
            for _ in range(50):
                time.sleep(0.1)
                try:
                    with socket.create_connection((config.HOST, config.PORT), timeout=0.5):
                        webbrowser.open(url)
                        return
                except OSError:
                    continue
        threading.Thread(target=_open_later, daemon=True).start()

    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
