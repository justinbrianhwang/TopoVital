import copy

import pytest

from topovital.util import PROJECT_ROOT, load_yaml

MITDB = PROJECT_ROOT / "data" / "raw" / "mit-bih-arrhythmia-database-1.0.0"
MITDB_RECORDS = ("100 101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 119 121 "
                 "122 123 124 200 201 202 203 205 207 208 209 210 212 213 214 215 217 219 220 221 "
                 "222 223 228 230 231 232 233 234").split()

requires_data = pytest.mark.skipif(not (MITDB / "RECORDS").exists(),
                                   reason="real MIT-BIH data not extracted (run: python -m topovital fetch)")


@pytest.fixture
def cfg():
    return copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "stage0.yaml"))
