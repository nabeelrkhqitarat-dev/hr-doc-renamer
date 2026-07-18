# hr-doc-renamer

Automatically rename scanned HR PDFs to a consistent convention by reading each
document with a **local** [Ollama](https://ollama.com) vision model. No cloud, no
API keys, no data ever leaves your machine.

```
scan_20260708115233.pdf   ->   PP_5018_Molla.pdf
scan_20260708115516.pdf   ->   AL_5018_Molla_2025.pdf
scan_20260708115910.pdf   ->   Offer_5018_Molla.pdf
```

## The naming convention

```
{DocType}_{EmployeeID}_{Name}[_{Year}].pdf
```

| Code    | Document                         | Year suffix |
|---------|----------------------------------|-------------|
| `AL`    | Annual Leave application         | yes (e.g. `_2025`) |
| `PP`    | Passport                         | no          |
| `HC`    | Health Card                      | no          |
| `QID`   | Qatar ID / residence permit      | no          |
| `Offer` | Offer letter                     | no          |
| `Contr` | Employment contract              | no          |

Codes, descriptions, the filename pattern, and the year rule are all defined in
[`config.example.yaml`](config.example.yaml) — edit them to fit your own scheme.

## Easiest way: the web portal 🌐

For non-technical users there is a simple browser portal — no command line needed:

```bash
python app.py            # or double-click start_portal.bat on Windows
```

Your browser opens `http://localhost:8010` with a 3-step wizard:

1. **Choose folder** — drag & drop the folder of scans (or click to browse)
2. **Reading documents** — a progress bar while the AI reads each scan
3. **Check & download** — review the proposed names (every name is editable,
   uncertain ones are highlighted), then download everything renamed as a ZIP

Originals are never modified — you always get a renamed copy, so a mistake is
never destructive. Colleagues on the same office network can use it too, at
`http://<your-pc-name>:8010`, while the processing stays on your machine.

## How it works

For each PDF the tool:

1. Renders the first page(s) to an image with **PyMuPDF** (also grabs any embedded
   text layer as an extra hint).
2. Sends the image to your local **Ollama** vision model and asks it, in strict
   JSON, for the document type, the person's name, the employee/staff number, and
   the year.
3. Maps the answer to a filename using your convention, then renames the file.

Everything runs offline against `http://localhost:11434`.

## Requirements

- **Python 3.9+**
- **[Ollama](https://ollama.com/download)** running locally, with a vision model:
  ```bash
  ollama pull qwen2.5vl:7b     # recommended for document OCR (Arabic + English)
  # lighter alternatives:  qwen2.5vl:3b  ·  granite3.2-vision:2b  ·  moondream
  ```
  > Note: `llama3.2-vision` (the older `mllama` architecture) no longer loads in
  > recent Ollama builds — use one of the models above.

## Install

```bash
git clone https://github.com/<you>/hr-doc-renamer.git
cd hr-doc-renamer
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml     # optional: customise your scheme
```

## Usage

The tool is **safe by default**: with no flags it does a *dry run* — it shows
exactly what it would rename and writes a CSV report, but changes nothing.

```bash
# 1. dry run — review the plan
python rename_docs.py "C:\path\to\scans"

# 2. happy with it? apply the renames
python rename_docs.py "C:\path\to\scans" --apply
```

| Flag           | Meaning                                                    |
|----------------|------------------------------------------------------------|
| `--apply`      | actually rename (default is a dry run)                     |
| `--config`     | use a specific config file                                 |
| `--recursive`  | also process PDFs in sub-folders                           |
| `--model`      | override the Ollama model for this run                     |
| `--report`     | path for the CSV report (default `rename_report.csv`)      |
| `--skip-named` | skip files that already match the convention              |

### The report

Every run writes `rename_report.csv` with one row per file: the original name, the
proposed name, the extracted fields, the model's confidence, and a status:

- `planned` / `renamed` — a new name was produced
- `review` — low confidence, unknown type, or a missing required field; **left
  untouched** for you to handle manually
- `unchanged` — already correctly named
- `error` — the file could not be processed (reason in `notes`)

## Notes & limitations

- **Employee ID** is only filled in when it is actually printed on the document
  (offer letters, contracts, and leave forms usually carry it; passports and QID
  cards do not). When it is missing, the tool uses the `unknown_id_placeholder`
  (default `XXXX`) so you can spot and fix it quickly — see the report.
- **Name spelling** is taken from the document. Use `name_overrides` in the config
  to normalise inconsistent transliterations (e.g. `Mullah` → `Molla`).
- Accuracy depends on scan quality and the chosen model. Always review the dry-run
  plan before using `--apply`.
- Real HR documents are private — the sample PDFs here are synthetic, and
  `.gitignore` keeps `samples/*.pdf` and reports out of git.

## License

MIT — see [LICENSE](LICENSE).
