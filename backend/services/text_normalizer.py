"""
Text normalization utilities for matching only.

IMPORTANT:
- This does NOT modify stored clause text.
- This is used only for query-time matching.
"""
from __future__ import annotations

import re
import string

OCR_REPLACEMENTS = {
    "lerminateg": "terminate",
    "terminateg": "terminate",
    "nolce": "notice",
    "agreernent": "agreement",
    "cornpensation": "compensation",
}


def normalize_for_match(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # fix common OCR errors
    for bad, good in OCR_REPLACEMENTS.items():
        text = text.replace(bad, good)

    # fix split tokens like "t ermi nate"
    text = re.sub(r"t\s*e\s*r\s*m\s*i\s*n\s*a\s*t\s*e", "terminate", text)

    return text.strip()


def detect_ocr_noise(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    if any(bad in lowered for bad in OCR_REPLACEMENTS.keys()):
        return True
    if re.search(r"t\s*e\s*r\s*m\s*i\s*n\s*a\s*t\s*e", lowered):
        return True
    return False
