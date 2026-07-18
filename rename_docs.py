#!/usr/bin/env python3
"""
hr-doc-renamer
==============
Rename scanned HR PDFs to a consistent convention by reading each document
with a local Ollama vision model. Nothing leaves your machine.

Convention (configurable):   {DocType}_{EmployeeID}_{Name}[_{Year}].pdf
  e.g.  scan_20260708115233.pdf  ->  PP_5018_Molla.pdf
        scan_20260708115516.pdf  ->  AL_5018_Molla_2025.pdf

Usage
-----
  # 1. dry run - shows what WOULD happen, changes nothing, writes a report
  python rename_docs.py "C:\\path\\to\\scans"

  # 2. once the plan looks right, apply it
  python rename_docs.py "C:\\path\\to\\scans" --apply

Options
-------
  --apply              actually rename files (default is a safe dry run)
  --config PATH        config file (default: config.yaml, else config.example.yaml)
  --recursive          also process PDFs in sub-folders
  --model NAME         override the Ollama model from the config
  --report PATH        where to write the CSV report (default: rename_report.csv)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from datetime import datetime

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing dependency 'pymupdf'. Run:  python -m pip install -r requirements.txt")
try:
    import requests
except ImportError:
    sys.exit("Missing dependency 'requests'. Run:  python -m pip install -r requirements.txt")
try:
    import yaml
except ImportError:
    sys.exit("Missing dependency 'pyyaml'. Run:  python -m pip install -r requirements.txt")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(explicit_path: str | None) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [explicit_path] if explicit_path else [
        os.path.join(here, "config.yaml"),
        os.path.join(here, "config.example.yaml"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            cfg["_path"] = path
            return cfg
    sys.exit("No config file found. Expected config.yaml or config.example.yaml.")


# --------------------------------------------------------------------------- #
# PDF -> images + text
# --------------------------------------------------------------------------- #
def pdf_pages(path: str, dpi: int, max_pages: int):
    """Return (list_of_png_bytes, extracted_text) for the first `max_pages`."""
    images, texts = [], []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi)
            images.append(pix.tobytes("png"))
            texts.append(page.get_text("text"))
    return images, "\n".join(t.strip() for t in texts if t.strip())


# --------------------------------------------------------------------------- #
# Prompt + Ollama call
# --------------------------------------------------------------------------- #
def build_prompt(cfg: dict, embedded_text: str) -> str:
    lines = ["You are an HR document filing assistant. Look at the scanned document",
             "image(s) and identify what it is and whom it belongs to.",
             "",
             "Classify the document as exactly ONE of these type codes:"]
    for dt in cfg["doc_types"]:
        aliases = ", ".join(dt.get("aliases", []))
        lines.append(f'  - "{dt["code"]}" = {dt["desc"]}' + (f" (e.g. {aliases})" if aliases else ""))
    lines += [
        '  - "UNKNOWN" if it matches none of the above or you are unsure.',
        "",
        "Then extract, when visible on the document:",
        "  - family_name: the person's surname / last name (single word).",
        "  - full_name: the person's full name as printed.",
        "  - employee_id: the company/employee/staff number if printed (NOT the",
        "                 passport or QID number). Empty string if not shown.",
        "  - year: the 4-digit year the document applies to (e.g. the leave year",
        "          on a leave form). Empty string if not applicable.",
        "  - confidence: your confidence from 0.0 to 1.0 in the type + name.",
        "",
        "Respond with ONLY a JSON object with these exact keys:",
        '{"doc_type":"","family_name":"","full_name":"","employee_id":"","year":"","confidence":0.0}',
    ]
    if embedded_text:
        snippet = embedded_text[:1500]
        lines += ["", "Text already extracted from the PDF (may be empty for scans):",
                  '"""', snippet, '"""']
    return "\n".join(lines)


def call_ai(cfg: dict, images_png: list[bytes], prompt: str) -> dict:
    """Dispatch to the configured AI backend.

    Provider is chosen by the AI_PROVIDER env var ("ollama" or "gemini").
    If unset, Gemini is used when GEMINI_API_KEY is present, else local Ollama.
    """
    provider = (os.environ.get("AI_PROVIDER") or "").strip().lower()
    if not provider:
        provider = "gemini" if os.environ.get("GEMINI_API_KEY") else "ollama"
    if provider == "gemini":
        return call_gemini(cfg, images_png, prompt)
    return call_ollama(cfg, images_png, prompt)


_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_gemini_model_cache: str | None = None  # first model name that worked


def _discover_gemini_model(api_key: str, timeout: int) -> str:
    """Ask the API which models this key can use and pick the best flash one.

    Keeps the tool working when Google renames or retires model ids.
    """
    resp = requests.get(f"{_GEMINI_BASE}/models",
                        headers={"x-goog-api-key": api_key},
                        params={"pageSize": 1000}, timeout=timeout)
    resp.raise_for_status()
    names = [m["name"].split("/")[-1] for m in resp.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    blocked = ("embedding", "image", "tts", "live", "audio", "veo", "aqa", "thinking")
    flashes = [n for n in names
               if "flash" in n and not any(b in n for b in blocked)]
    if not flashes:
        raise RuntimeError(f"No usable Gemini model found; key sees: {names[:20]}")

    def rank(name: str) -> tuple:
        m = re.search(r"(\d+(?:\.\d+)?)", name)
        version = float(m.group(1)) if m else 0.0
        return (
            "lite" not in name,               # full flash beats flash-lite
            "preview" not in name and "exp" not in name,  # stable beats preview
            version,                          # newest version
            "latest" in name,                 # -latest alias as tiebreak
        )
    return max(flashes, key=rank)


def call_gemini(cfg: dict, images_png: list[bytes], prompt: str) -> dict:
    """Cloud backend: Google Gemini (free tier works). Needs GEMINI_API_KEY."""
    global _gemini_model_cache
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    timeout = cfg["ollama"].get("timeout_seconds", 300)

    candidates = [c for c in (
        _gemini_model_cache,
        os.environ.get("GEMINI_MODEL", "").strip() or None,
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ) if c]

    payload = {
        "contents": [{"parts": [{"text": prompt}] + [
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.b64encode(img).decode("ascii")}}
            for img in images_png
        ]}],
        "generationConfig": {"temperature": 0,
                             "response_mime_type": "application/json"},
    }

    resp = None
    for attempt, model in enumerate(dict.fromkeys(candidates)):
        resp = requests.post(f"{_GEMINI_BASE}/models/{model}:generateContent",
                             headers={"x-goog-api-key": api_key},
                             json=payload, timeout=timeout)
        if resp.status_code == 404:      # model id unknown -> try the next one
            continue
        resp.raise_for_status()
        _gemini_model_cache = model
        break
    else:
        # every known name 404'd -> ask the API what this key can actually use
        model = _discover_gemini_model(api_key, timeout)
        resp = requests.post(f"{_GEMINI_BASE}/models/{model}:generateContent",
                             headers={"x-goog-api-key": api_key},
                             json=payload, timeout=timeout)
        resp.raise_for_status()
        _gemini_model_cache = model

    data = resp.json()
    try:
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response: {str(data)[:200]}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return JSON: {raw[:200]}")


def call_ollama(cfg: dict, images_png: list[bytes], prompt: str) -> dict:
    o = cfg["ollama"]
    payload = {
        "model": cfg.get("_model_override") or o["model"],
        "prompt": prompt,
        "images": [base64.b64encode(img).decode("ascii") for img in images_png],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": o.get("temperature", 0),
            # vision inputs are token-hungry; the Ollama default (4096) is too small
            "num_ctx": o.get("num_ctx", 8192),
        },
    }
    resp = requests.post(
        o["host"].rstrip("/") + "/api/generate",
        json=payload,
        timeout=o.get("timeout_seconds", 300),
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return JSON: {raw[:200]}")


# --------------------------------------------------------------------------- #
# Build the target filename
# --------------------------------------------------------------------------- #
def sanitize(text: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]', "", text or "").strip()
    return re.sub(r"\s+", " ", text)


def pick_name(result: dict, cfg: dict) -> str:
    naming = cfg["naming"]
    if naming.get("name_part", "family") == "full":
        raw = result.get("full_name") or result.get("family_name") or ""
        name = re.sub(r"\s+", "", sanitize(raw))
    else:
        raw = result.get("family_name") or (result.get("full_name") or "").split()[-1:] or ""
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        name = sanitize(raw).replace(" ", "")
    overrides = {k.lower(): v for k, v in (naming.get("name_overrides") or {}).items()}
    return overrides.get(name.lower(), name)


def type_needs_year(code: str, cfg: dict) -> bool:
    for dt in cfg["doc_types"]:
        if dt["code"].lower() == code.lower():
            return bool(dt.get("year"))
    return False


def canonical_code(code: str, cfg: dict) -> str | None:
    for dt in cfg["doc_types"]:
        if dt["code"].lower() == (code or "").lower():
            return dt["code"]
    return None


def build_filename(result: dict, cfg: dict) -> tuple[str | None, str]:
    """Return (filename_without_ext or None, note)."""
    naming = cfg["naming"]
    code = canonical_code(result.get("doc_type", ""), cfg)
    if code is None:
        return None, f"unrecognised doc_type '{result.get('doc_type')}'"

    name = pick_name(result, cfg)
    if not name:
        return None, "no name detected"

    emp_id = sanitize(str(result.get("employee_id") or "")).replace(" ", "")
    note = ""
    if not emp_id:
        emp_id = naming.get("unknown_id_placeholder", "XXXX")
        note = "employee id not on document -> placeholder"

    base = naming["pattern"].format(doc_type=code, employee_id=emp_id, name=name, year="")
    if type_needs_year(code, cfg):
        year = re.sub(r"\D", "", str(result.get("year") or ""))[:4]
        if len(year) == 4:
            base += naming.get("year_suffix", "_{year}").format(year=year)
        else:
            note = (note + "; " if note else "") + "year required but not detected"
    return base, note


def resolve_collision(folder: str, base: str, ext: str, taken: set[str]) -> str:
    candidate = base + ext
    if candidate.lower() not in taken and not os.path.exists(os.path.join(folder, candidate)):
        return candidate
    i = 2
    while True:
        candidate = f"{base}_v{i}{ext}"
        if candidate.lower() not in taken and not os.path.exists(os.path.join(folder, candidate)):
            return candidate
        i += 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def find_pdfs(folder: str, recursive: bool) -> list[str]:
    out = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            out += [os.path.join(root, f) for f in files if f.lower().endswith(".pdf")]
    else:
        out = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    return sorted(out)


def already_named(fname: str, cfg: dict) -> bool:
    """Skip files that already look like our convention (start with a known code)."""
    stem = os.path.splitext(fname)[0]
    for dt in cfg["doc_types"]:
        if stem.lower().startswith(dt["code"].lower() + "_"):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Rename scanned HR PDFs via a local Ollama vision model.")
    ap.add_argument("folder", help="folder containing the PDFs to rename")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    ap.add_argument("--config", help="path to config file")
    ap.add_argument("--recursive", action="store_true", help="descend into sub-folders")
    ap.add_argument("--model", help="override the Ollama model name")
    ap.add_argument("--report", default="rename_report.csv", help="CSV report path")
    ap.add_argument("--skip-named", action="store_true",
                    help="skip files already matching the convention")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        sys.exit(f"Not a folder: {args.folder}")

    cfg = load_config(args.config)
    if args.model:
        cfg["_model_override"] = args.model
    min_conf = cfg.get("min_confidence", 0.45)

    pdfs = find_pdfs(args.folder, args.recursive)
    if not pdfs:
        print("No PDF files found.")
        return 0

    mode = "APPLY" if args.apply else "DRY RUN"
    model = cfg.get("_model_override") or cfg["ollama"]["model"]
    print(f"[{mode}] {len(pdfs)} PDF(s)  |  model={model}  |  config={os.path.basename(cfg['_path'])}\n")

    rows = []
    taken: set[str] = set()
    for path in pdfs:
        folder = os.path.dirname(path)
        orig = os.path.basename(path)
        row = {"original": orig, "proposed": "", "doc_type": "", "name": "",
               "employee_id": "", "year": "", "confidence": "", "status": "", "notes": ""}

        if args.skip_named and already_named(orig, cfg):
            row["status"], row["notes"] = "skipped", "already named"
            print(f"  skip   {orig}  (already named)")
            rows.append(row)
            continue

        try:
            images, text = pdf_pages(path, cfg["render"]["dpi"], cfg["render"]["max_pages"])
            if not images:
                raise ValueError("no pages rendered")
            result = call_ai(cfg, images, build_prompt(cfg, text))
        except Exception as exc:  # noqa: BLE001  - report per-file, keep going
            row["status"], row["notes"] = "error", str(exc)[:200]
            print(f"  ERROR  {orig}  ->  {row['notes']}")
            rows.append(row)
            continue

        conf = float(result.get("confidence") or 0)
        row.update(doc_type=result.get("doc_type", ""),
                   name=pick_name(result, cfg),
                   employee_id=result.get("employee_id", ""),
                   year=result.get("year", ""),
                   confidence=f"{conf:.2f}")

        base, note = build_filename(result, cfg)
        if base is None:
            row["status"], row["notes"] = "review", note
            print(f"  review {orig}  ->  {note}")
            rows.append(row)
            continue
        if conf < min_conf:
            row["status"] = "review"
            row["notes"] = f"low confidence {conf:.2f} < {min_conf}"
            row["proposed"] = base + ".pdf"
            print(f"  review {orig}  ->  {base}.pdf  (low confidence {conf:.2f})")
            rows.append(row)
            continue

        target = resolve_collision(folder, base, ".pdf", taken)
        taken.add(target.lower())
        row["proposed"] = target
        row["notes"] = note

        if target == orig:
            row["status"] = "unchanged"
            print(f"  ok     {orig}  (already correct)")
        elif args.apply:
            os.rename(path, os.path.join(folder, target))
            row["status"] = "renamed"
            print(f"  RENAME {orig}  ->  {target}" + (f"   [{note}]" if note else ""))
        else:
            row["status"] = "planned"
            print(f"  plan   {orig}  ->  {target}" + (f"   [{note}]" if note else ""))
        rows.append(row)

    with open(args.report, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_ren = sum(1 for r in rows if r["status"] in ("renamed", "planned"))
    n_rev = sum(1 for r in rows if r["status"] == "review")
    n_err = sum(1 for r in rows if r["status"] == "error")
    verb = "renamed" if args.apply else "to rename"
    print(f"\nSummary: {n_ren} {verb}, {n_rev} need review, {n_err} errors.")
    print(f"Report: {os.path.abspath(args.report)}")
    if not args.apply and n_ren:
        print("Re-run with --apply to perform the renames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
