"""Out-of-fold U-Net horizon forecasts for the train windows: each window takes the forecast of fc_fold<d % 4>, the model that never saw its day.

Reads cache_top3/<panel>/train_causalfill_fc{0..3}.npz (same window order), writes train_causalfill_fcoof.npz.
Evaluation splits keep one file per fold model (QUEUE_FC_EVAL=_fc<k> in tfb.queue.final, probabilities averaged afterwards).
Run: python -m tfb.queue.fc_oof [--folds 4] [--prefix _fc] [--out _fcoof]
"""

from __future__ import annotations

import argparse

import numpy as np

from tfb.common import CACHE, QUEUE_PANELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--prefix", default="_fc")
    ap.add_argument("--out", default="_fcoof")
    a = ap.parse_args()
    for pn in QUEUE_PANELS:
        zs = [np.load(CACHE / pn / f"train_causalfill{a.prefix}{k}.npz") for k in range(a.folds)]
        d, T = zs[0]["d"], zs[0]["T"]
        for z in zs[1:]:
            assert np.array_equal(z["d"], d) and np.array_equal(z["T"], T), pn
        fold = d % a.folds
        out = {"d": d, "T": T}
        for key in zs[0].files:
            if key in ("d", "T"):
                continue
            arr = np.array(zs[0][key])  # (n, ...) per-window arrays
            for k in range(1, a.folds):
                sel = fold == k
                arr[sel] = zs[k][key][sel]
            out[key] = arr
        np.savez(CACHE / pn / f"train_causalfill{a.out}.npz", **out)
        print(f"{pn}: {len(d)} windows, per fold {np.bincount(fold, minlength=a.folds).tolist()}")


if __name__ == "__main__":
    main()
