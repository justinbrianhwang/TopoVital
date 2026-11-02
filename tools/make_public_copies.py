"""Create path-sanitized publication copies of the Regime 1 audit CSVs."""
from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import re
import socket
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topovital.util import atomic_write_json, atomic_write_text, sha256_file, utc_now

DATA_DIR = ROOT / "outputs" / "stage1_features" / "development"
PATH_COLUMNS = ("job_path", "pre_reference")
SOURCE_NAMES = ("regime1_ledger.csv", "regime1_injection_log.csv")
VERBATIM_NAMES = (
    "snippet_allocation.csv",
    "regime1_summary.json",
    "hashes/regime1_full_run_prerun.json",
)
RULE = "Keep the component beginning with outputs; replace path separators with '/'."


def repo_relative_path(value: str) -> str:
    """Discard the machine-specific prefix through the component before outputs."""
    if not value or value.strip().casefold() == "nan":
        return ""
    components = re.split(r"[\\/]", value)
    positions = [i for i, component in enumerate(components) if component.casefold() == "outputs"]
    if not positions:
        raise ValueError(f"Path has no outputs component: {value!r}")
    suffix = components[positions[-1]:]
    if any(component in ("", ".", "..") for component in suffix):
        raise ValueError(f"Unsafe outputs path: {value!r}")
    return "/".join(["outputs", *suffix[1:]])


def public_frame(source: pd.DataFrame) -> pd.DataFrame:
    public = source.copy(deep=True)
    for column in PATH_COLUMNS:
        if column in public:
            public[column] = public[column].map(repo_relative_path)
    return public


def residual_hits(frame: pd.DataFrame, *, user_names: tuple[str, ...] | None = None,
                  host_name: str | None = None) -> list[str]:
    """Return cell locations that still contain a local path or identity."""
    names = (getpass.getuser(), os.environ.get("USERNAME", ""), os.environ.get("USER", ""))
    identities = {name.casefold() for name in (user_names if user_names is not None else names)
                  if name}
    host = socket.gethostname() if host_name is None else host_name
    if host:
        identities.add(host.casefold())
    patterns = (re.compile(r"[A-Za-z]:[\\/]", re.IGNORECASE),
                re.compile(r"/Users/|/home/|\\Users\\", re.IGNORECASE))
    hits = []
    for row_number, row in enumerate(frame.itertuples(index=False, name=None), start=2):
        for column, value in zip(frame.columns, row):
            cell = str(value)
            if any(pattern.search(cell) for pattern in patterns) or any(
                identity in cell.casefold() for identity in identities
            ):
                hits.append(f"{column}:row_{row_number}")
    return hits


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    atomic_write_text(path, buffer.getvalue())


def make_public_copies(root: Path = ROOT) -> dict:
    data_dir = root / "outputs" / "stage1_features" / "development"
    public_dir = data_dir / "public"
    files = []
    source_hashes = {}
    for source_name in SOURCE_NAMES:
        source_path = data_dir / source_name
        original_hash = sha256_file(source_path)
        source_hashes[source_path] = original_hash
        source = pd.read_csv(source_path, dtype=str, keep_default_na=False, encoding="utf-8")
        missing = set(PATH_COLUMNS) - set(source.columns)
        if missing:
            raise ValueError(f"Missing path columns in {source_path}: {sorted(missing)}")
        public = public_frame(source)
        public_path = public_dir / source_name.replace(".csv", ".public.csv")
        write_csv(public_path, public)
        saved = pd.read_csv(public_path, dtype=str, keep_default_na=False, encoding="utf-8")
        if list(saved.columns) != list(source.columns) or len(saved) != len(source) or any(
            not saved[column].equals(source[column])
            for column in source.columns if column not in PATH_COLUMNS
        ):
            raise ValueError(f"Non-path CSV content changed in {public_path}")
        hits = residual_hits(saved)
        if hits:
            raise ValueError(f"Residual identifiers in {public_path}: {', '.join(hits)}")
        files.append({
            "source_path": source_path.relative_to(root).as_posix(),
            "source_sha256": original_hash,
            "public_path": public_path.relative_to(root).as_posix(),
            "public_sha256": sha256_file(public_path),
            "rows": len(saved),
            "columns": len(saved.columns),
            "column_names": list(saved.columns),
            "rewritten_columns": [name for name in PATH_COLUMNS if name in saved],
            "rule": RULE,
            "residual_check": "passed",
        })
    published_verbatim = []
    for name in VERBATIM_NAMES:
        path = data_dir / name
        published_verbatim.append({"path": path.relative_to(root).as_posix(),
                                   "sha256": sha256_file(path)})
    for path, original_hash in source_hashes.items():
        if sha256_file(path) != original_hash:
            raise RuntimeError(f"Source changed during publication: {path}")
    manifest = {
        "generated_utc": utc_now(),
        "files": files,
        "published_verbatim": published_verbatim,
        "note": "Original audit files are kept unchanged locally and are not published.",
    }
    atomic_write_json(public_dir / "PUBLIC_MANIFEST.json", manifest)
    return manifest


def make_level3_public_copy(root: Path = ROOT) -> dict:
    data_dir = root / "outputs" / "stage1_analysis" / "development" / "level3_features"
    public_dir = data_dir / "public"
    source_path = data_dir / "waveform_regeneration_check.csv"
    public_path = public_dir / "waveform_regeneration_check.public.csv"
    original_hash = sha256_file(source_path)
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, encoding="utf-8")
    if "job_path" not in source.columns:
        raise ValueError(f"Missing path columns in {source_path}: ['job_path']")
    public = source.copy(deep=True)
    public["job_path"] = public["job_path"].map(repo_relative_path)
    write_csv(public_path, public)
    saved = pd.read_csv(public_path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(saved.columns) != list(source.columns) or len(saved) != len(source) or any(
        not saved[column].equals(source[column])
        for column in source.columns if column != "job_path"
    ):
        raise ValueError(f"Non-path CSV content changed in {public_path}")
    hits = residual_hits(saved)
    if hits:
        raise ValueError(f"Residual identifiers in {public_path}: {', '.join(hits)}")
    file = {
        "source_path": source_path.relative_to(root).as_posix(),
        "source_sha256": original_hash,
        "public_path": public_path.relative_to(root).as_posix(),
        "public_sha256": sha256_file(public_path),
        "rows": len(saved),
        "columns": len(saved.columns),
        "column_names": list(saved.columns),
        "rewritten_columns": ["job_path"],
        "rule": RULE,
        "residual_check": "passed",
    }
    if sha256_file(source_path) != original_hash:
        raise RuntimeError(f"Source changed during publication: {source_path}")
    manifest = {
        "generated_utc": utc_now(),
        "files": [file],
        "published_verbatim": [],
        "note": "Original audit files are kept unchanged locally and are not published.",
    }
    atomic_write_json(public_dir / "PUBLIC_MANIFEST.json", manifest)
    return manifest


def _public_value(value, root: Path):
    if isinstance(value, dict):
        public = {}
        for key, item in value.items():
            public_key = _public_value(key, root)
            if public_key in public:
                raise ValueError(f"Public dict keys collide after path rewrite: {public_key!r}")
            public[public_key] = _public_value(item, root)
        return public
    if isinstance(value, list):
        return [_public_value(item, root) for item in value]
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    prefix = str(root.resolve()).replace("\\", "/").rstrip("/")
    if normalized.casefold().startswith((prefix + "/").casefold()):
        return normalized[len(prefix) + 1:]
    if re.match(r"^(?:[A-Za-z]:/|//|/)", normalized) and "/outputs/" in normalized.casefold():
        return repo_relative_path(value)
    return value


def make_final_test_public_copy(root: Path = ROOT) -> dict:
    """Publish sanitized Development verification reports and their replay ledgers."""
    root = Path(root)
    base = root / "outputs/final_test"
    public_dir = base / "public"
    entries = {}
    for directory in sorted(base.glob("dev_verification*")):
        if not directory.is_dir():
            continue
        for name in ("dev_verification_report.json", "replay/jobs_ledger.csv"):
            source = directory / name
            original_hash = sha256_file(source)
            target = public_dir / directory.name / name
            if source.suffix == ".json":
                content = json.loads(source.read_text(encoding="utf-8"))
                atomic_write_json(target, _public_value(content, root))
            else:
                frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8")
                for column in frame.columns:
                    frame[column] = frame[column].map(lambda cell: _public_value(cell, root))
                write_csv(target, frame)
            root_prefix = re.sub(r"[\\/]+", "/", str(root.resolve())).casefold()
            public_text = re.sub(r"[\\/]+", "/", target.read_text(encoding="utf-8")).casefold()
            if root_prefix in public_text:
                raise RuntimeError(f"Residual project root in public file: {target}")
            if sha256_file(source) != original_hash:
                raise RuntimeError(f"Source changed during publication: {source}")
            entries[target.relative_to(root).as_posix()] = {
                "source_path": source.relative_to(root).as_posix(),
                "public_sha256": sha256_file(target),
                "source_sha256": original_hash}
    if not entries:
        raise ValueError("No Development verification reports found")
    manifest = {"files": entries}
    atomic_write_json(public_dir / "PUBLIC_MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level3", action="store_true", help="Publish the Level 3 waveform check only")
    parser.add_argument("--final-test", action="store_true", help="Publish final Test Development verification")
    args = parser.parse_args(argv)
    if args.level3 and args.final_test:
        parser.error("Choose only one publication mode")
    try:
        result = (make_final_test_public_copy() if args.final_test else
                  make_level3_public_copy() if args.level3 else make_public_copies())
        noun = "files" if args.final_test else "CSVs"
        print(f"Wrote {len(result['files'])} public {noun} and PUBLIC_MANIFEST.json")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Public copy failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
