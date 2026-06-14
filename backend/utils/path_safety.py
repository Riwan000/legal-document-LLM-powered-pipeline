"""
Path safety utilities to prevent path traversal (CWE-22 / SonarQube S2083).

User-controlled file paths (e.g. from upload forms or API form fields) must be
validated to ensure they resolve *inside* an approved base directory before they
are opened. A path such as ``../../etc/passwd`` or an absolute path pointing
elsewhere on disk is rejected.
"""
import os
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]


class UnsafePathError(ValueError):
    """Raised when a path resolves outside every permitted base directory."""


def resolve_within(candidate: PathLike, allowed_bases: Iterable[PathLike]) -> Path:
    """
    Resolve ``candidate`` and verify it stays inside one of ``allowed_bases``.

    The path is fully resolved (symlinks and ``..`` segments collapsed) and then
    checked against each allowed base. The first base that contains the resolved
    path wins; if none do, the path is rejected.

    Args:
        candidate: The (possibly user-controlled) path to validate.
        allowed_bases: Directories the resolved path must live under.

    Returns:
        The fully-resolved, validated ``Path``.

    Raises:
        UnsafePathError: If the resolved path is outside every allowed base.
    """
    resolved = Path(os.path.realpath(candidate))
    for base in allowed_bases:
        base_resolved = Path(os.path.realpath(base))
        try:
            resolved.relative_to(base_resolved)
            return resolved
        except ValueError:
            continue
    raise UnsafePathError(
        f"Path escapes permitted directories: {candidate!r}"
    )


def is_within(candidate: PathLike, allowed_bases: Iterable[PathLike]) -> bool:
    """Return ``True`` if ``candidate`` safely resolves inside an allowed base."""
    try:
        resolve_within(candidate, allowed_bases)
        return True
    except UnsafePathError:
        return False
