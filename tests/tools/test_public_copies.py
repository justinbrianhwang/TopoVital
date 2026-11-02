import json
import os
from pathlib import Path

import pandas as pd
import pytest

import tools.make_public_copies as public_copies
from tools.make_public_copies import (make_level3_public_copy, make_public_copies, public_frame,
                                      repo_relative_path, residual_hits)
from topovital.util import sha256_file


def test_windows_path_with_unicode_user_becomes_posix_repo_path():
    path = r"C:\Users\Zoë_Ünal\Desktop\TopoVital\outputs\stage1_features\job.npz"
    assert repo_relative_path(path) == "outputs/stage1_features/job.npz"
    assert repo_relative_path("") == ""
    assert repo_relative_path("NaN") == ""
    with pytest.raises(ValueError, match="no outputs component"):
        repo_relative_path(r"C:\Users\Zoë_Ünal\job.npz")


def test_public_csv_preserves_other_strings_and_sources(tmp_path: Path):
    data = tmp_path / "outputs" / "stage1_features" / "development"
    data.mkdir(parents=True)
    csv_text = (
        "job_path,pre_reference,scale_a,other\n"
        "C:\\Users\\Zoë_Ünal\\TopoVital\\outputs\\stage1_features\\development\\regime1_jobs\\a.npz,"
        "C:\\Users\\Zoë_Ünal\\TopoVital\\outputs\\stage1_features\\development\\jobs\\b.npz,"
        "1.0000000000000002e-15,alpha\n"
    )
    for name in ("regime1_ledger.csv", "regime1_injection_log.csv"):
        (data / name).write_text(csv_text, encoding="utf-8")
    (data / "snippet_allocation.csv").write_text("event_id\na\n", encoding="utf-8")
    (data / "regime1_summary.json").write_text("{}\n", encoding="utf-8")
    (data / "hashes").mkdir()
    (data / "hashes" / "regime1_full_run_prerun.json").write_text("{}\n", encoding="utf-8")
    before = {name: sha256_file(data / name)
              for name in ("regime1_ledger.csv", "regime1_injection_log.csv")}

    manifest = make_public_copies(tmp_path)

    assert len(manifest["files"]) == 2
    for name in before:
        assert sha256_file(data / name) == before[name]
        public = data / "public" / name.replace(".csv", ".public.csv")
        rows = pd.read_csv(public, dtype=str, keep_default_na=False)
        assert rows.loc[0, "job_path"] == (
            "outputs/stage1_features/development/regime1_jobs/a.npz")
        assert rows.loc[0, "pre_reference"] == (
            "outputs/stage1_features/development/jobs/b.npz")
        assert rows.loc[0, "scale_a"] == "1.0000000000000002e-15"
        assert rows.loc[0, "other"] == "alpha"
    saved_manifest = json.loads((data / "public" / "PUBLIC_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(saved_manifest["published_verbatim"]) == 3


def test_residual_check_catches_paths_and_host_name():
    frame = pd.DataFrame({"clean": ["outputs/a.npz", "host=unit-host", "C:/Users/jane/a"],
                          "other": ["ok", "ok", "ok"]})
    assert residual_hits(frame, user_names=("unit-user",), host_name="unit-host") == [
        "clean:row_3", "clean:row_4"]
    assert residual_hits(pd.DataFrame({"x": [r"\\Users\\jane\\a"]}),
                         user_names=("unit-user",), host_name="unit-host")


def test_only_path_columns_are_rewritten():
    source = pd.DataFrame({"other": ["1.0000000000000002e-15"],
                           "job_path": [r"D:\repo\outputs\x.npz"],
                           "pre_reference": [""]}, dtype=str)
    public = public_frame(source)
    assert list(public.columns) == list(source.columns)
    assert public.loc[0, "other"] == source.loc[0, "other"]
    assert public.loc[0, "job_path"] == "outputs/x.npz"


def test_level3_public_copy_preserves_other_columns_and_source(tmp_path: Path):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    data = tmp_path / "outputs" / "stage1_analysis" / "development" / "level3_features"
    data.mkdir(parents=True)
    source_path = data / "waveform_regeneration_check.csv"
    source_path.write_text(
        "job_path,passed,notes,pre_reference\n"
        "C:\\Users\\Zoë_Ünal\\TopoVital\\outputs\\stage1_analysis\\development\\level3\\a.npz,True,,reference-a\n"
        "C:\\Users\\Zoë_Ünal\\TopoVital\\outputs\\stage1_analysis\\development\\level3\\b.npz,False,checked,\n",
        encoding="utf-8",
    )
    original_bytes = source_path.read_bytes()

    manifest = make_level3_public_copy(tmp_path)

    public_path = data / "public" / "waveform_regeneration_check.public.csv"
    source = pd.read_csv(source_path, dtype=str, keep_default_na=False, encoding="utf-8")
    saved = pd.read_csv(public_path, dtype=str, keep_default_na=False, encoding="utf-8")
    assert saved["job_path"].tolist() == [
        "outputs/stage1_analysis/development/level3/a.npz",
        "outputs/stage1_analysis/development/level3/b.npz",
    ]
    assert saved.drop(columns="job_path").equals(source.drop(columns="job_path"))
    assert saved["passed"].tolist() == ["True", "False"]
    assert saved["notes"].tolist() == ["", "checked"]
    assert source_path.read_bytes() == original_bytes

    saved_manifest = json.loads((data / "public" / "PUBLIC_MANIFEST.json").read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert saved_manifest["generated_utc"]
    assert saved_manifest["published_verbatim"] == []
    assert saved_manifest["note"] == "Original audit files are kept unchanged locally and are not published."
    assert saved_manifest["files"] == [{
        "source_path": source_path.relative_to(tmp_path).as_posix(),
        "source_sha256": sha256_file(source_path),
        "public_path": public_path.relative_to(tmp_path).as_posix(),
        "public_sha256": sha256_file(public_path),
        "rows": 2,
        "columns": 4,
        "column_names": ["job_path", "passed", "notes", "pre_reference"],
        "rewritten_columns": ["job_path"],
        "rule": public_copies.RULE,
        "residual_check": "passed",
    }]


def test_cli_defaults_to_regime1_and_level3_flag_selects_level3(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(public_copies, "make_public_copies", lambda: calls.append("regime1") or {"files": [1, 2]})
    monkeypatch.setattr(public_copies, "make_level3_public_copy", lambda: calls.append("level3") or {"files": [1]})

    public_copies.main([])
    assert calls == ["regime1"]
    assert "Wrote 2 public CSVs" in capsys.readouterr().out

    public_copies.main(["--level3"])
    assert calls == ["regime1", "level3"]
    assert "Wrote 1 public CSVs" in capsys.readouterr().out
