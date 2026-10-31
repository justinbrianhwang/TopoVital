# Execution-resource settings (not scientific settings): pin BLAS/OpenMP to one
# thread before numpy is imported anywhere, to avoid nested parallelism.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from topovital.cli import main  # noqa: E402

raise SystemExit(main())
