"""Data acquisition from the official PhysioNet zip distributions + integrity check.

The zips (data/*.zip) are the PhysioNet "Download the ZIP file" artefacts. Files are
extracted (never overwritten if already present and verified) and every file listed
in the distribution's own SHA256SUMS.txt is verified.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from topovital.util import resolve, sha256_file


class IntegrityError(RuntimeError):
    pass


def _parse_sha256sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip().lstrip("*")] = digest.lower()
    return out


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    """Extract members not already present; refuse path traversal. Returns #extracted."""
    n = 0
    dest_r = dest.resolve()
    for m in zf.infolist():
        target = (dest / m.filename).resolve()
        if not str(target).startswith(str(dest_r)):
            raise IntegrityError(f"unsafe path in zip: {m.filename}")
        if m.is_dir() or target.exists():
            continue
        zf.extract(m, dest)
        n += 1
    return n


def fetch(cfg: dict) -> tuple[pd.DataFrame, dict]:
    dest = resolve(cfg["data"]["extract_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    rows, info = [], {}
    used_nstdb = set(cfg["data"]["nstdb"]["noise_records"])
    for key in ("mitdb", "nstdb"):
        d = cfg["data"][key]
        zpath = resolve(d["zip"])
        if not zpath.exists():
            raise FileNotFoundError(
                f"{zpath} not found. Download the official ZIP from {d['url']} (no credentials "
                f"required) into {zpath.parent} and re-run: python -m topovital fetch")
        zip_sha = sha256_file(zpath)
        with zipfile.ZipFile(zpath) as zf:
            n_extracted = _safe_extract(zf, dest)
        root = dest / d["root"]
        sums = _parse_sha256sums(root / "SHA256SUMS.txt")
        records = (root / "RECORDS").read_text().split()
        mism = []
        for name, expected in sorted(sums.items()):
            fp = root / name
            actual = sha256_file(fp) if fp.exists() else None
            ok = actual == expected
            if not ok:
                mism.append(name)
            stem = Path(name).stem
            if key == "mitdb":
                used = "/" not in name and stem in records and Path(name).suffix in (".hea", ".dat", ".atr")
            else:
                used = name in {f"{r}.{e}" for r in used_nstdb for e in ("hea", "dat")}
            rows.append({"dataset": key, "dataset_version": d["version"], "source_url": d["url"],
                         "source_zip": zpath.name, "relative_path": f"{d['root']}/{name}",
                         "sha256_expected": expected, "sha256_actual": actual,
                         "verified": ok, "used_in_stage0": used})
        rows.append({"dataset": key, "dataset_version": d["version"], "source_url": d["url"],
                     "source_zip": zpath.name, "relative_path": zpath.name,
                     "sha256_expected": None, "sha256_actual": zip_sha,
                     "verified": None, "used_in_stage0": True})
        info[key] = {"zip_sha256": zip_sha, "files_listed": len(sums), "mismatches": mism,
                     "newly_extracted": n_extracted, "n_records_file": len(records),
                     "license": d["license"]}
        if mism:
            raise IntegrityError(f"{key}: SHA256 mismatch for {mism[:10]}")
    return pd.DataFrame(rows), info
