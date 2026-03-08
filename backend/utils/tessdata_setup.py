import os
import shutil
import urllib.request
from pathlib import Path

_ARA_URL = (
    "https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata"
)


def _find_tessdata_dir() -> Path | None:
    """Return the Tesseract tessdata directory or None if Tesseract is not installed."""
    candidates = [
        Path(os.environ.get("TESSDATA_PREFIX", "")) / "tessdata",
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return None


def check_and_download():
    tessdata = _find_tessdata_dir()
    if tessdata is None:
        return  # Tesseract not installed; OCR will fail anyway
    target = tessdata / "ara.traineddata"
    if target.exists():
        return  # Already present
    try:
        print("[tessdata] Downloading Arabic language pack …")
        with urllib.request.urlopen(_ARA_URL, timeout=60) as resp, \
             open(target, "wb") as f:
            shutil.copyfileobj(resp, f)
        print(f"[tessdata] Saved to {target}")
    except Exception as e:
        print(f"[tessdata] Warning: could not download ara.traineddata: {e}")
