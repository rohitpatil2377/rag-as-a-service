# RAG as a Service — OCR & Document Extraction

How the platform turns any uploaded file into indexed text, which engine reads what, and how
to enable the stronger PaddleOCR engine. All engines are **sovereign** (self-hosted, on-prem).

## 1. Auto-routing: the right extractor per file (FR-B4)

Ingestion never uses "one model for everything". `svc_doc_intelligence/classify.py` identifies
each file from its **MIME type and its magic bytes** (so a mislabeled `octet-stream` still routes
correctly), and `service._route` sends it down the best path:

| Kind (detected) | Path | Engine |
|---|---|---|
| Native PDF (has a text layer) | direct extract — perfect, no OCR | pdfplumber / pypdf |
| Scanned PDF (no text layer) | rasterise 300 DPI → OCR | Tesseract / PaddleOCR |
| Image (png/jpg/tiff/webp/bmp…) | OCR | Tesseract / PaddleOCR |
| DOCX | zipped-XML extract | stdlib + defusedxml |
| HTML | visible-text extract | html.parser |
| Text | decode | — |

A PDF that *looks* digital but yields almost no text is auto-reclassified as a scan and sent to OCR
(never indexed empty). The kind is logged (`doc_classified`) for every ingest.

## 2. OCR engines (choose with `RAG_OCR_BACKEND`)

| Backend | Reads well | CPU? | Notes |
|---|---|---|---|
| `noop` (default) | — | — | Refuses scans with a clear 422 (no silent empty index) |
| `tesseract` | clean/moderate printed text, incl. Indic (`eng+hin`) | ✅ | In the base image. Preprocessing + real confidence (below) |
| `paddleocr` | **low-quality scans, receipts, rotated & multilingual/Indic** | ✅ | PP-OCRv4, optional extra — the recommended upgrade |
| `trocr` | **handwriting** / scene text | GPU recommended | Optional; Tesseract/Paddle are printed-text only |

**Handwriting and logo/display art** (e.g. a stylised masthead) are **not** readable by Tesseract or
PaddleOCR — those are printed-text engines. Handwriting needs `trocr` (or a vision-language model);
logo art is effectively a graphic, not text.

## 3. Preprocessing & confidence (Tesseract path)

Each page is cleaned before OCR — the biggest CPU accuracy lever on faint/small/skewed scans:
EXIF auto-orient → grayscale → **upscale small scans** to ~300 DPI-equivalent → autocontrast →
optional OpenCV deskew (if `cv2` present). It is **gentle by design**: we do *not* hard-binarize or
blur by default, because that destroys thin strokes and *hurts* clean-ish scans (measured: a readable
image regressed 11→2 tokens with binarization, and lifted 11→15 with the gentle path). Hard Otsu
binarization is opt-in via `RAG_OCR_BINARIZE=true` for the rare noisy bitonal fax.

Confidence is now the **real** mean per-word score from Tesseract (`image_to_data`), not a hardcoded
guess — pages below `RAG_OCR_MIN_CONFIDENCE` are surfaced as low-confidence for human review.

Config: `RAG_OCR_PREPROCESS` (default on), `RAG_OCR_BINARIZE` (off), `RAG_OCR_PSM`/`RAG_OCR_OEM`,
`RAG_OCR_MIN_CONFIDENCE`, `RAG_OCR_LANG` (e.g. `eng+hin`).

## 4. Enabling PaddleOCR (CPU, sovereign)

PaddleOCR (PP-OCRv4) is markedly better than Tesseract on the *hard* printed cases — faded thermal
receipts, halftone print, rotated text, dense multilingual pages — and runs on CPU. It is an
**optional extra** (paddlepaddle is large), off by default.

**Build it into the image:**
```bash
docker build --build-arg INSTALL_PADDLEOCR=true -f apps/api/Dockerfile -t ragaas-api:paddle .
```
Then run with `RAG_OCR_BACKEND=paddleocr` (and `RAG_OCR_LANG=eng` → mapped to Paddle `en`;
`hin` → `devanagari`).

**Deployment constraints (learned the hard way — read before rolling out):**
- **Pin 2.7.x.** paddleocr ≥ 2.8 pulls `albumentations/albucore/stringzilla`, which need a C toolchain.
  The 2.7.x line installs from wheels only.
- **`setuptools` is required** at runtime (paddle imports it) — installed by the build arg.
- **Match the wheel to the host.** The stock `paddlepaddle` wheel targets specific glibc/AVX levels; on a
  bleeding-edge base OS (e.g. glibc 2.41) it can crash on import (`munmap_chunk`). Use a paddlepaddle
  wheel/base image built for the **target** host's glibc + CPU (the official `paddlepaddle/paddle`
  images pin a compatible OS), or the noavx build on CPUs without AVX2.
- **Air-gapped / sovereign:** PaddleOCR fetches its det/rec/cls models on first use (a one-time
  bootstrap, like the TEI/llama model pulls). Pre-mirror them and point `RAG_OCR_MODEL_DIR` at a
  folder with `det/`, `rec/`, `cls/` subfolders so nothing is fetched at runtime.

The engine code (`PaddleOCREngine`) and its line-reconstruction/confidence logic are in the base image
and unit-tested (`reconstruct_paddle_page`); only the heavy runtime wheel is opt-in.

## 5. Roadmap

- **TrOCR (handwriting)** — wire `RAG_OCR_BACKEND=trocr` on a GPU model host (the "gpu wala OCR").
- **Per-document engine selection** — the classifier could pick Tesseract vs PaddleOCR vs TrOCR per
  detected kind/quality automatically (today it's one configured backend for all scans).
