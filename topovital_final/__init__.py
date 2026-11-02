"""Frozen, one-shot final Test evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path


def final_code_hash(package_dir: Path | None = None) -> str:
    """Hash final Python sources in filename order with normalized line endings."""
    digest = hashlib.sha256()
    for path in sorted((package_dir or Path(__file__).parent).glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()
