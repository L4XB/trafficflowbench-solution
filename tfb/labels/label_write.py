"""Materialise a label speed field version (see tfb.labels.label_oracle.fields) as cache_top3/<panel>/train_speed_<name>.npy.

Run: ORACLE_VERSIONS=mix0.6 python -m tfb.labels.label_write --version mix0.6 --name mix
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from tfb.common import CACHE, QUEUE_PANELS
from tfb.labels.label_oracle import fields


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    os.environ["ORACLE_VERSIONS"] = a.version
    for pn in QUEUE_PANELS:
        f = fields(pn)[a.version].astype(np.float32)
        np.save(CACHE / pn / f"train_speed_{a.name}.npy", f)
        print(pn, "written", f.shape, flush=True)


if __name__ == "__main__":
    main()
