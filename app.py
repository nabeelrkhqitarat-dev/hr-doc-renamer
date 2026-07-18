#!/usr/bin/env python3
"""
hr-doc-renamer web portal
=========================
A tiny local web app on top of rename_docs.py: drop a folder of scanned HR
PDFs in the browser, watch them get identified, review/fix the proposed
names, and download everything renamed as a ZIP. Runs fully offline against
your local Ollama.

Run:
    python app.py            # then open http://localhost:8010
or double-click start_portal.bat on Windows.
"""
from __future__ import annotations

import base64
import hmac
import os
import queue
import re
import shutil
import threading
import time
import uuid
import webbrowser
import zipfile

try:
    import uvicorn
    from fastapi import FastAPI, File, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    raise SystemExit("Missing web dependencies. Run:  python -m pip install -r requirements.txt")

import rename_docs as rd

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(HERE, ".jobs")
JOB_TTL_SECONDS = 2 * 60 * 60  # forget jobs after 2 hours
PORT = int(os.environ.get("PORT", "8010"))

CFG = rd.load_config(None)
app = FastAPI(title="hr-doc-renamer portal")
if os.path.isdir(os.path.join(HERE, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# Optional shared access code (set the PORTAL_PASSCODE env var to enable).
PASSCODE = os.environ.get("PORTAL_PASSCODE", "").strip()


def _authorized(request: Request) -> bool:
    if not PASSCODE:
        return True
    supplied = (request.headers.get("x-portal-code")
                or request.query_params.get("code") or "")
    return hmac.compare_digest(supplied, PASSCODE)


def _deny() -> JSONResponse:
    return JSONResponse({"error": "access code required"}, status_code=401)


# --------------------------------------------------------------------------- #
# Local AI worker bridge
# --------------------------------------------------------------------------- #
# A machine running Ollama can register as a "worker": it long-polls
# /api/worker/task, runs the vision model locally, and posts the result back.
# The portal then prefers the worker (no rate limits) and falls back to the
# cloud AI when no worker has polled recently. Enabled by setting the
# WORKER_SECRET env var on the server and the same value on the worker.
WORKER_SECRET = os.environ.get("WORKER_SECRET", "").strip()
WORKER_TASK_TIMEOUT = int(os.environ.get("WORKER_TASK_TIMEOUT", "300"))
WORKER_TASKS: dict[str, dict] = {}
WORKER_QUEUE: "queue.Queue[str]" = queue.Queue()
_worker_last_seen = 0.0


def _worker_online() -> bool:
    return bool(WORKER_SECRET) and (time.time() - _worker_last_seen) < 90


def _worker_auth(request: Request) -> bool:
    if not WORKER_SECRET:
        return False
    return hmac.compare_digest(request.headers.get("x-worker-secret", ""), WORKER_SECRET)


def _run_ai(images: list[bytes], prompt: str) -> tuple[dict, str]:
    """Run one extraction, preferring the local worker. Returns (result, engine)."""
    if _worker_online():
        task_id = uuid.uuid4().hex
        done = threading.Event()
        WORKER_TASKS[task_id] = {"prompt": prompt, "images": images,
                                 "event": done, "result": None, "error": None}
        WORKER_QUEUE.put(task_id)
        if done.wait(timeout=WORKER_TASK_TIMEOUT):
            task = WORKER_TASKS.pop(task_id)
            if task["error"]:
                raise RuntimeError(f"local AI worker: {task['error']}")
            return task["result"], "local"
        WORKER_TASKS.pop(task_id, None)  # worker vanished mid-task -> fall back
    return rd.call_ai(CFG, images, prompt), "cloud"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _cleanup_old_jobs() -> None:
    now = time.time()
    with JOBS_LOCK:
        stale = [jid for jid, j in JOBS.items() if now - j["created"] > JOB_TTL_SECONDS]
        for jid in stale:
            shutil.rmtree(JOBS[jid]["dir"], ignore_errors=True)
            del JOBS[jid]


def _safe_basename(name: str) -> str:
    """Keep only the final path component and strip forbidden characters."""
    name = name.replace("\\", "/").split("/")[-1]
    return re.sub(r'[<>:"|?*\x00-\x1f]', "", name).strip() or "file.pdf"


def _process_job(job_id: str) -> None:
    job = JOBS[job_id]
    for entry in job["files"]:
        entry["status"] = "processing"
        path = os.path.join(job["dir"], "in", entry["stored"])
        try:
            if rd.already_named(entry["original"], CFG):
                entry["status"] = "named"
                entry["proposed"] = entry["original"]
                entry["note"] = "already matches the naming convention"
                continue
            images, text = rd.pdf_pages(path, CFG["render"]["dpi"], CFG["render"]["max_pages"])
            if not images:
                raise ValueError("could not read any page from this PDF")
            result, entry["engine"] = _run_ai(images, rd.build_prompt(CFG, text))
            conf = float(result.get("confidence") or 0)
            base, note, suggested = rd.build_filename(result, CFG)
            entry["confidence"] = round(conf, 2)
            entry["doc_type"] = rd.canonical_code(result.get("doc_type", ""), CFG) or result.get("doc_type", "")
            if base is None:
                entry["status"] = "review"
                entry["proposed"] = entry["original"]
                entry["note"] = note
            elif suggested:
                entry["status"] = "suggest"
                entry["proposed"] = base + ".pdf"
                entry["note"] = note
            else:
                entry["proposed"] = base + ".pdf"
                entry["note"] = note
                entry["status"] = "review" if conf < CFG.get("min_confidence", 0.45) else "ok"
                if entry["status"] == "review" and not note:
                    entry["note"] = "low confidence - please double-check"
        except Exception as exc:  # noqa: BLE001 - report per file, keep going
            entry["status"] = "error"
            entry["proposed"] = entry["original"]
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                entry["note"] = ("The AI service is busy (free quota limit). "
                                 "This file was left unchanged - try it again in a few minutes.")
            else:
                entry["note"] = msg[:200]
        finally:
            job["done"] += 1
    job["status"] = "review"


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/ping")
def ping(request: Request):
    if not _authorized(request):
        return _deny()
    if _worker_online():
        engine = "worker"
    elif (os.environ.get("AI_PROVIDER") or "").lower() == "gemini" or (
            not os.environ.get("AI_PROVIDER") and os.environ.get("GEMINI_API_KEY")):
        engine = "gemini"
    else:
        engine = "ollama"
    return {"ok": True, "protected": bool(PASSCODE),
            "worker": _worker_online(), "engine": engine}


@app.get("/api/worker/task")
def worker_task(request: Request, wait: int = 20):
    """Long-poll endpoint for the local AI worker. 204 = nothing to do."""
    global _worker_last_seen
    if not _worker_auth(request):
        return JSONResponse({"error": "bad worker secret"}, status_code=403)
    _worker_last_seen = time.time()
    try:
        task_id = WORKER_QUEUE.get(timeout=max(1, min(wait, 28)))
    except queue.Empty:
        return JSONResponse(None, status_code=204)
    task = WORKER_TASKS.get(task_id)
    if task is None:  # timed out and reclaimed before the worker got it
        return JSONResponse(None, status_code=204)
    _worker_last_seen = time.time()
    return {"task_id": task_id, "prompt": task["prompt"],
            "images": [base64.b64encode(i).decode("ascii") for i in task["images"]]}


class WorkerResult(BaseModel):
    task_id: str
    result: dict | None = None
    error: str | None = None


@app.post("/api/worker/result")
def worker_result(body: WorkerResult, request: Request):
    global _worker_last_seen
    if not _worker_auth(request):
        return JSONResponse({"error": "bad worker secret"}, status_code=403)
    _worker_last_seen = time.time()
    task = WORKER_TASKS.get(body.task_id)
    if task is None:  # too late - the server already fell back
        return {"ok": False, "note": "task expired"}
    task["result"] = body.result
    task["error"] = body.error if body.result is None else None
    task["event"].set()
    return {"ok": True}


@app.post("/api/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)):
    if not _authorized(request):
        return _deny()
    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(os.path.join(job_dir, "in"), exist_ok=True)

    entries = []
    for i, up in enumerate(files):
        original = _safe_basename(up.filename or f"file_{i}.pdf")
        if not original.lower().endswith(".pdf"):
            continue
        stored = f"{i:04d}.pdf"  # avoid any name collisions on disk
        with open(os.path.join(job_dir, "in", stored), "wb") as fh:
            shutil.copyfileobj(up.file, fh)
        entries.append({
            "original": original, "stored": stored, "proposed": "",
            "doc_type": "", "confidence": None, "status": "waiting", "note": "",
            "engine": "",
        })

    if not entries:
        shutil.rmtree(job_dir, ignore_errors=True)
        return JSONResponse({"error": "No PDF files found in that folder."}, status_code=400)

    job = {"id": job_id, "dir": job_dir, "files": entries, "done": 0,
           "status": "processing", "created": time.time(), "zip": None}
    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(target=_process_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "total": len(entries)}


@app.get("/api/job/{job_id}")
def job_status(job_id: str, request: Request):
    if not _authorized(request):
        return _deny()
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {
        "status": job["status"], "done": job["done"], "total": len(job["files"]),
        "files": [{k: v for k, v in e.items() if k != "stored"} for e in job["files"]],
    }


class FinalizeItem(BaseModel):
    original: str
    final_name: str
    include: bool = True


class FinalizeBody(BaseModel):
    items: list[FinalizeItem]


@app.post("/api/job/{job_id}/finalize")
def finalize(job_id: str, body: FinalizeBody, request: Request):
    if not _authorized(request):
        return _deny()
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "unknown job"}, status_code=404)

    by_original = {e["original"]: e for e in job["files"]}
    zip_path = os.path.join(job["dir"], "renamed.zip")
    used: set[str] = set()
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in body.items:
            entry = by_original.get(item.original)
            if entry is None or not item.include:
                continue
            name = _safe_basename(item.final_name)
            if not name.lower().endswith(".pdf"):
                name += ".pdf"
            stem, ext = os.path.splitext(name)
            n, final = 2, name
            while final.lower() in used:  # keep names unique inside the zip
                final = f"{stem}_v{n}{ext}"
                n += 1
            used.add(final.lower())
            zf.write(os.path.join(job["dir"], "in", entry["stored"]), final)
            count += 1
    if count == 0:
        os.remove(zip_path)
        return JSONResponse({"error": "Nothing selected to download."}, status_code=400)
    job["zip"] = zip_path
    return {"count": count}


@app.get("/api/job/{job_id}/download")
def download(job_id: str, request: Request):
    if not _authorized(request):
        return _deny()
    job = JOBS.get(job_id)
    if not job or not job.get("zip") or not os.path.isfile(job["zip"]):
        return JSONResponse({"error": "nothing to download"}, status_code=404)
    return FileResponse(job["zip"], filename="renamed_documents.zip",
                        media_type="application/zip")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "portal.html"), "r", encoding="utf-8") as fh:
        return fh.read()


def main() -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)
    in_cloud = os.path.exists("/.dockerenv") or bool(os.environ.get("RENDER") or os.environ.get("SPACE_ID"))
    if not in_cloud:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    print(f"\n  hr-doc-renamer portal  ->  http://localhost:{PORT}\n"
          f"  (colleagues on your network can use http://<this-pc-name>:{PORT})\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
