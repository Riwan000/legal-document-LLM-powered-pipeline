"""
Contract profile loader (FIX 2).

Loads YAML contract profiles from backend/contract_profiles/ and caches them.
Invalid or missing profile leads to structured workflow failure; no expected
clause logic is hardcoded in Python.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

try:
    import yaml
except ImportError:
    yaml = None

# Default profile directory relative to backend package
_BACKEND_DIR = Path(__file__).resolve().parent.parent
CONTRACT_PROFILES_DIR = _BACKEND_DIR / "contract_profiles"

_log = logging.getLogger(__name__)

# Process-level cache: contract_type_key -> profile dict
_profile_cache: Dict[str, Dict[str, Any]] = {}


def load_contract_profile(contract_type: str) -> Dict[str, Any]:
    """
    Load a contract profile by type. Profiles are cached.

    Args:
        contract_type: Contract type key (e.g. 'nda', 'employment', 'msa').
                       Case-insensitive; normalized to lowercase for file lookup.

    Returns:
        Dict with keys: contract_type, expected_clauses, optional_clauses, risk_weights.

    Raises:
        FileNotFoundError: If no profile file exists for the type.
        ValueError: If the YAML is invalid or missing required keys.
    """
    key = contract_type.strip().lower()
    if not key:
        raise ValueError("contract_type cannot be empty")

    if key in _profile_cache:
        return _profile_cache[key]

    if yaml is None:
        raise RuntimeError("PyYAML is required to load contract profiles. Install with: pip install PyYAML")

    path = CONTRACT_PROFILES_DIR / f"{key}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Contract profile not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid contract profile: expected YAML mapping, got {type(data)}")

    expected = data.get("expected_clauses")
    if expected is None:
        raise ValueError("Contract profile must define 'expected_clauses'")
    if not isinstance(expected, list):
        raise ValueError("expected_clauses must be a list")
    if not all(isinstance(c, str) for c in expected):
        raise ValueError("expected_clauses must be a list of strings")

    optional = data.get("optional_clauses")
    if optional is None:
        optional = []
    if not isinstance(optional, list):
        raise ValueError("optional_clauses must be a list")
    if not all(isinstance(c, str) for c in optional):
        raise ValueError("optional_clauses must be a list of strings")

    risk_weights = data.get("risk_weights") or {}
    if not isinstance(risk_weights, dict):
        raise ValueError("risk_weights must be a mapping")

    profile = {
        "contract_type": data.get("contract_type", contract_type),
        "expected_clauses": [c.strip().lower() for c in expected],
        "optional_clauses": [c.strip().lower() for c in optional],
        "risk_weights": {k.strip().lower(): v for k, v in risk_weights.items()},
    }
    _profile_cache[key] = profile
    return profile
