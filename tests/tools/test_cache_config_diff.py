import json

import numpy as np
import pandas as pd
import yaml

from tools.regime1_cache_config_diff import compare, differing_keys, flatten_config
from topovital.util import config_hash


def test_flatten_and_diff_find_only_approval_flip():
    current = {"regime1": {"approved_for_full_run": True,
                            "noise": {"types": ["bw", "ma", "em"]}},
               "stage1": {"embedding": {"m": 5}}}
    validation = {"regime1": {"approved_for_full_run": False,
                               "noise": {"types": ["bw", "ma", "em"]}},
                  "stage1": {"embedding": {"m": 5}}}
    assert flatten_config(current)["stage1.embedding.m"] == 5
    assert differing_keys(current, validation) == ["regime1.approved_for_full_run"]


def test_compare_reconstructs_both_hashes_and_reads_cached_metadata(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    cfg = {"approved_for_full_run": True, "noise": {"types": ["bw"]}}
    stage1 = {"embedding": {"m": 5}}
    (configs / "stage1_regime1.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (configs / "stage1.yaml").write_text(yaml.safe_dump(stage1), encoding="utf-8")
    validation = {**cfg, "approved_for_full_run": False}
    current_hash = config_hash({"regime1": cfg, "stage1": stage1})
    validation_hash = config_hash({"regime1": validation, "stage1": stage1})
    data = tmp_path / "outputs" / "stage1_features" / "development"
    jobs = data / "regime1_jobs"
    jobs.mkdir(parents=True)
    job = jobs / "job.npz"
    np.savez_compressed(job, metadata=np.array(json.dumps({"cache_key": "unchanged"})))
    ledger_path = r"C:\Users\옛사용자\TopoVital\outputs\stage1_features\development\regime1_jobs\job.npz"
    rows = [{"config_hash": validation_hash, "cached": "True",
             "cache_key": "unchanged", "job_path": ledger_path} for _ in range(108)]
    rows.append({"config_hash": current_hash, "cached": "False",
                 "cache_key": "other", "job_path": ledger_path})
    pd.DataFrame(rows).to_csv(data / "regime1_ledger.csv", index=False)

    result = compare(tmp_path)

    assert result["reproduced"] == {current_hash: True, validation_hash: True}
    assert result["n_cached"] == result["cache_keys_match"] == 108
    assert result["differing_keys"] == ["regime1.approved_for_full_run"]
    assert (data / "hashes" / "regime1_cache_config_diff.json").is_file()
