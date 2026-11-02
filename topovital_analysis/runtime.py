"""Analysis runtime checks and result version history.

Version 1 is every analysis code state up to and including code hash
2f6a358682ca000d504cc44545654c21d7ebc4a73826407d72730875fe1ccd33.
Its thread variables were ineffective, so runs used the machine default unless
they were exported in the shell.

Results produced with version 1 are reproduced by version 2 only if they ran
with 1 thread. The Level 3 full run did; Level 2 and the factorial runs did not.
"""

import os

from threadpoolctl import threadpool_info

from topovital.cohort import BlockedError


ANALYSIS_CODE_VERSION = "2.0.0"
VERSION_1_LAST_CODE_HASH = "2f6a358682ca000d504cc44545654c21d7ebc4a73826407d72730875fe1ccd33"
EXPECTED_BLAS_THREADS = 1

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
_CHECKED_APIS = {"openblas", "openmp", "mkl", "blis"}


def blas_thread_report():
    """Return the configured thread variables and loaded native threadpools."""
    return {
        "env": {name: os.environ.get(name) for name in _THREAD_ENV_VARS},
        "threadpools": [(item["internal_api"], item["num_threads"])
                        for item in threadpool_info()],
    }


def verify_blas_threads(expected=EXPECTED_BLAS_THREADS):
    """Block analysis if environment or loaded native libraries use other counts."""
    report = blas_thread_report()
    wrong = [f"{name}={value!r}" for name, value in report["env"].items()
             if value != str(expected)]
    wrong.extend(f"{api}={count}" for api, count in report["threadpools"]
                 if api in _CHECKED_APIS and count != expected)
    if wrong:
        raise BlockedError(f"BLAS thread check failed (expected {expected}): {', '.join(wrong)}")
    return report
