"""One-job VR worker. Input: .npy on stdin. Output: compact JSON on stdout."""
from __future__ import annotations

import json
import base64
import io
import sys
import argparse

import numpy as np

from topovital.util import peak_rss_mb

from .ph import vr_diagrams, vr_persistence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagrams")
    args = parser.parse_args()
    points = np.load(io.BytesIO(base64.b64decode(sys.stdin.buffer.read())), allow_pickle=False)
    if args.diagrams:
        result = vr_diagrams(points)
        np.savez_compressed(args.diagrams, h0=result.pop("h0"), h1=result.pop("h1"))
    else:
        result = vr_persistence(points)
    result["child_peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
