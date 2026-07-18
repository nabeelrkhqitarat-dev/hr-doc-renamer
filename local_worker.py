#!/usr/bin/env python3
"""
Local AI worker for the hr-doc-renamer portal
=============================================
Run this on a machine with Ollama + a vision model installed. It connects
OUT to the cloud portal (no open ports, no tunnels needed), picks up
documents queued by the portal, reads them with the local model, and sends
the extraction result back.

While this worker is running, the portal uses your local AI (unlimited,
free) instead of the cloud AI. Stop it, and the portal falls back to the
cloud AI automatically.

Usage:
    set PORTAL_URL=https://your-app.onrender.com
    set WORKER_SECRET=the-same-secret-you-set-on-the-server
    python local_worker.py

or double-click start_worker.bat (edit the two values inside it first).
"""
from __future__ import annotations

import base64
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rename_docs as rd  # noqa: E402

PORTAL_URL = (os.environ.get("PORTAL_URL") or "").strip().rstrip("/")
WORKER_SECRET = (os.environ.get("WORKER_SECRET") or "").strip()


def main() -> int:
    if not PORTAL_URL or not WORKER_SECRET:
        print("Set the PORTAL_URL and WORKER_SECRET environment variables first.")
        return 1

    cfg = rd.load_config(None)
    model = cfg["ollama"]["model"]
    headers = {"X-Worker-Secret": WORKER_SECRET}
    print(f"\n  Local AI worker\n  portal : {PORTAL_URL}\n  model  : {model} (Ollama)\n"
          f"  Waiting for documents... press Ctrl+C to stop.\n")

    errors_in_a_row = 0
    while True:
        try:
            r = requests.get(f"{PORTAL_URL}/api/worker/task",
                             headers=headers, params={"wait": 25}, timeout=40)
            if r.status_code == 403:
                print("  ! The portal rejected the WORKER_SECRET - check that it matches.")
                return 1
            if r.status_code != 200:
                errors_in_a_row = 0
                continue
            errors_in_a_row = 0
            task = r.json()
            images = [base64.b64decode(i) for i in task["images"]]
            t0 = time.time()
            try:
                result = rd.call_ollama(cfg, images, task["prompt"])
                payload = {"task_id": task["task_id"], "result": result}
                print(f"  done  {task['task_id'][:8]}  "
                      f"{result.get('doc_type', '?'):<10} {time.time() - t0:.0f}s")
            except Exception as exc:  # noqa: BLE001 - report failure to server
                payload = {"task_id": task["task_id"], "error": str(exc)[:300]}
                print(f"  FAIL  {task['task_id'][:8]}  {str(exc)[:80]}")
            requests.post(f"{PORTAL_URL}/api/worker/result",
                          headers=headers, json=payload, timeout=30)
        except KeyboardInterrupt:
            print("\n  Stopped. The portal will fall back to the cloud AI.")
            return 0
        except requests.RequestException as exc:
            errors_in_a_row += 1
            if errors_in_a_row in (1, 10):
                print(f"  ... connection problem ({str(exc)[:80]}), retrying")
            time.sleep(min(60, 5 * errors_in_a_row))


if __name__ == "__main__":
    raise SystemExit(main())
