import pytest

from conftest import MITDB, MITDB_RECORDS, requires_data
from topovital import cohort as C

# NOTE: seeds used below are test-only values to exercise the mechanism; they are not
# the protocol's split seed and no real assignment is derived from them.
TEST_ONLY_SEED = 12345


def test_cohort_counts_48_47_to_44_43(cfg):
    m = C.build_manifest(MITDB_RECORDS, cfg["cohort"])
    assert (len(m), m.subject_id.nunique()) == (48, 47)
    kept = m[~m.excluded]
    assert (len(kept), kept.subject_id.nunique()) == (44, 43)
    assert set(m[m.excluded].record_id) == {"102", "104", "107", "217"}
    assert m.set_index("record_id").loc[["201", "202"], "subject_id"].nunique() == 1


def test_cohort_assert_fires_on_wrong_counts(cfg):
    with pytest.raises(C.CohortError):
        C.build_manifest(MITDB_RECORDS[:-1], cfg["cohort"])


def test_split_requires_seed(cfg):
    with pytest.raises(C.BlockedError):
        C.split_subjects(["S100"], {**cfg["split"], "seed": None})


def test_split_subject_level_disjoint_and_201_202_together(cfg):
    m = C.build_manifest(MITDB_RECORDS, cfg["cohort"])
    kept = m[~m.excluded]
    sp = C.split_subjects(kept.subject_id.tolist(), {**cfg["split"], "seed": TEST_ONLY_SEED, "stratification": "none"})
    dev = {s for s, v in sp.items() if v == "development"}
    test = {s for s, v in sp.items() if v == "test"}
    assert (len(dev), len(test)) == (22, 21) and not dev & test
    rec_split = kept.assign(split=kept.subject_id.map(sp)).set_index("record_id").split
    assert rec_split["201"] == rec_split["202"]


def test_select_lead_by_name_regardless_of_order():
    assert C.select_lead(["MLII", "V1"], "MLII") == 0
    assert C.select_lead(["V5", "MLII"], "MLII") == 1
    with pytest.raises(C.CohortError):
        C.select_lead(["V5", "V2"], "MLII")


@requires_data
def test_real_record_114_mlii_is_channel_1(cfg):
    import wfdb

    assert C.select_lead(wfdb.rdheader(str(MITDB / "114")).sig_name, "MLII") == 1
    assert C.select_lead(wfdb.rdheader(str(MITDB / "100")).sig_name, "MLII") == 0
