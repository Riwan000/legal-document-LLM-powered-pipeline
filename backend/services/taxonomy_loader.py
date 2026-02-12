"""
Load taxonomy YAML files from config/taxonomies/.

Used for conduct indicators and benefit categories in contract review.
"""
from pathlib import Path
from typing import Dict, Any, List

try:
    import yaml
except ImportError:
    yaml = None

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
TAXONOMIES_DIR = _PROJECT_ROOT / "config" / "taxonomies"

# TODO(v0.3): Replace substring matching with word-boundary regex to avoid false
# positives (e.g. 'rule' vs 'ruler'). When implemented: re.search(rf"\b{re.escape(keyword)}\b", text).


def load_conduct_taxonomy() -> Dict[str, Any]:
    """Load config/taxonomies/conduct.yaml. Returns dict with 'conduct_indicators' list."""
    path = TAXONOMIES_DIR / "conduct.yaml"
    if not path.exists():
        return {"conduct_indicators": []}
    if yaml is None:
        return {"conduct_indicators": []}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {"conduct_indicators": []}


def load_benefits_taxonomy() -> Dict[str, List[str]]:
    """Load config/taxonomies/benefits.yaml. Returns dict mapping category name to list of keywords."""
    path = TAXONOMIES_DIR / "benefits.yaml"
    if not path.exists():
        return {}
    if yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    categories = data.get("benefit_categories")
    if not isinstance(categories, dict):
        return {}
    return {k: v if isinstance(v, list) else [] for k, v in categories.items()}
