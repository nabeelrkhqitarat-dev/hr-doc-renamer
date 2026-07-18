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

import os
import re
import shutil
import threading
import time
import uuid
import webbrowser
import zipfile

try:
    import uvicorn
    from fastapi import FastAPI, File, UploadFile
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
            result = rd.call_ollama(CFG, images, rd.build_prompt(CFG, text))
            conf = float(result.get("confidence") or 0)
            base, note = rd.build_filename(result, CFG)
            entry["confidence"] = round(conf, 2)
            entry["doc_type"] = rd.canonical_code(result.get("doc_type", ""), CFG) or result.get("doc_type", "")
            if base is None:
                entry["status"] = "review"
                entry["proposed"] = entry["original"]
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
            entry["note"] = str(exc)[:200]
        finally:
            job["done"] += 1
    job["status"] = "review"


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
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
def job_status(job_id: str):
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
def finalize(job_id: str, body: FinalizeBody):
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
def download(job_id: str):
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
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    print(f"\n  hr-doc-renamer portal  ->  http://localhost:{PORT}\n"
          f"  (colleagues on your network can use http://<this-pc-name>:{PORT})\n")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
