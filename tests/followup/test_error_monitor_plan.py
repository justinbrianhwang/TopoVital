from topovital_followup.error_monitor.plan import needed_basefits


def test_ten_subject_plan():
    subjects = [f"S{i}" for i in range(10)]
    pilot = needed_basefits(["S0"], subjects, [0.001, 0.01, 0.1, 1, 10])
    assert pilot.counts() == {"basefits_by_size": {7: 36, 8: 9, 9: 1},
                              "unique_basefits": 46, "unique_units": 650,
                              "monitor_fits": 112}
    full = needed_basefits(subjects, subjects, [0.001, 0.01, 0.1, 1, 10])
    assert full.counts()["monitor_fits"] == 1120
