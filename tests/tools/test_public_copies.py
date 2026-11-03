import json
import os
from pathlib import Path

import pandas as pd
import pytest

import tools.make_public_copies as public_copies
from tools.make_public_copies import (make_final_test_public_copy, make_level3_public_copy, make_public_copies, public_frame,
                                      repo_relative_path, residual_hits)
from topovital.util import sha256_file


def test_test_run_sanitizes_recursive_paths_and_merges_manifest(tmp_path):
    root = tmp_path / "Zo챘_횥nal" / "TopoVital"
    base = root / "outputs/final_test"
    (base / "hashes").mkdir(parents=True)
    public = base / "public"
    public.mkdir()
    existing = {"files": {"outputs/final_test/public/dev_verification/report.json": {"source_sha256": "keep"}}}
    (public / "PUBLIC_MANIFEST.json").write_text(json.dumps(existing), encoding="utf-8")
    private = str(root / "outputs/final_test/jobs/a.json")
    payload = {"nested": [{private: [r"D:\Python\bin\libopenblas.dll", "Zo챘_횥nal", "unit-host",
                                    "//server/share/report.json", "/home/person/lib/core.so"]}]}
    for name in public_copies.TEST_RUN_NAMES:
        path = base / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    before = {name: sha256_file(base / name) for name in public_copies.TEST_RUN_NAMES}
    manifest = public_copies.make_test_run_public_copy(root, user="Zo챘_횥nal", host="unit-host",
                                                         home=tmp_path / "Zo챘_횥nal")
    assert manifest["files"]["outputs/final_test/public/dev_verification/report.json"] == {"source_sha256": "keep"}
    for name in before:
        assert sha256_file(base / name) == before[name]
        data = json.loads((public / "test_run" / name).read_text(encoding="utf-8"))
        assert data == {"nested": [{"outputs/final_test/jobs/a.json": [
            "<external>/bin/libopenblas.dll", "<user>", "<host>",
            "<external>/share/report.json", "<external>/lib/core.so"]}]}


def test_test_run_residual_prevents_manifest_write(tmp_path, monkeypatch):
    base = tmp_path / "outputs/final_test"
    (base / "hashes").mkdir(parents=True)
    for name in public_copies.TEST_RUN_NAMES:
        (base / name).write_text('{"key":"clean"}', encoding="utf-8")
    monkeypatch.setattr(public_copies, "sanitize_test_value", lambda *a, **k: {"path": "Z:/private/a"})
    with pytest.raises(RuntimeError, match="Residual"):
        public_copies.make_test_run_public_copy(tmp_path, user="person", host="machine")
    assert not (base / "public/PUBLIC_MANIFEST.json").exists()


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

    monkeypatch.setattr(public_copies, "make_final_test_public_copy",
                        lambda: calls.append("final-test") or {"files": {"a": {}, "b": {}}})
    public_copies.main(["--final-test"])
    assert calls == ["regime1", "level3", "final-test"]
    assert "Wrote 2 public files" in capsys.readouterr().out


def test_final_test_public_copy_rewrites_unicode_root_paths(tmp_path):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    root = tmp_path / "Zo챘_횥nal" / "TopoVital"
    verification = root / "outputs/final_test/dev_verification_2"
    replay = verification / "replay"
    replay.mkdir(parents=True)
    private = str(root / "outputs/final_test/dev_verification_2/replay/jobs/a.npz")
    report = verification / "dev_verification_report.json"
    report.write_text(json.dumps({"checks": [{"value": {
        "path": private,
        "no_test_outputs": {"value": {"jobs_content": {private: "sha256"}}},
    }}]}), encoding="utf-8")
    ledger = replay / "jobs_ledger.csv"
    ledger.write_text(f"job_path,status\n{private},ok\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in (report, ledger)}

    manifest = make_final_test_public_copy(root)

    for path, content in before.items():
        assert path.read_bytes() == content
    public = root / "outputs/final_test/public/dev_verification_2"
    copied_report = json.loads((public / report.name).read_text(encoding="utf-8"))
    public_path = "outputs/final_test/dev_verification_2/replay/jobs/a.npz"
    assert copied_report["checks"][0]["value"]["path"] == public_path
    jobs_content = copied_report["checks"][0]["value"]["no_test_outputs"]["value"]["jobs_content"]
    assert jobs_content == {public_path: "sha256"}
    copied_ledger = pd.read_csv(public / "replay/jobs_ledger.csv")
    assert copied_ledger.job_path.iloc[0] == public_path
    assert len(manifest["files"]) == 2
    for public_path, entry in manifest["files"].items():
        assert sha256_file(root / entry["source_path"]) == entry["source_sha256"]
        assert sha256_file(root / public_path) == entry["public_sha256"]


def test_final_test_public_copy_rejects_rewritten_key_collision(tmp_path):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    root = tmp_path / "TopoVital"
    verification = root / "outputs/final_test/dev_verification"
    replay = verification / "replay"
    replay.mkdir(parents=True)
    private = str(root / "outputs/final_test/jobs/a.npz")
    relative = "outputs/final_test/jobs/a.npz"
    (verification / "dev_verification_report.json").write_text(
        json.dumps({"jobs_content": {private: "first", relative: "second"}}), encoding="utf-8")
    (replay / "jobs_ledger.csv").write_text("job_path\n", encoding="utf-8")

    with pytest.raises(ValueError, match="keys collide"):
        make_final_test_public_copy(root)

    assert not (root / "outputs/final_test/public/PUBLIC_MANIFEST.json").exists()


@pytest.mark.parametrize("source_name", ["dev_verification_report.json", "replay/jobs_ledger.csv"])
def test_final_test_public_copy_rejects_residual_root(tmp_path, source_name):
    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    root = tmp_path / "TopoVital"
    verification = root / "outputs/final_test/dev_verification"
    replay = verification / "replay"
    replay.mkdir(parents=True)
    private = str(root / "outputs/final_test/jobs/a.npz").swapcase()
    report = verification / "dev_verification_report.json"
    ledger = replay / "jobs_ledger.csv"
    report.write_text(json.dumps({"note": "prefix=" + private}), encoding="utf-8")
    ledger.write_text("note\nprefix=" + private.replace("\\", "/") + "\n", encoding="utf-8")
    if source_name == "dev_verification_report.json":
        ledger.write_text("note\nclean\n", encoding="utf-8")
    else:
        report.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Residual project root"):
        make_final_test_public_copy(root)

    public = root / "outputs/final_test/public"
    assert (public / "dev_verification" / source_name).exists()
    assert not (public / "PUBLIC_MANIFEST.json").exists()
