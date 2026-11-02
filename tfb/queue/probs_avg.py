"""Average per-window probability dumps (probs.npz from tfb.queue.final --dump-probs) over several runs.

Used for the fold-model forecasts: one final run per fc_fold<k> evaluation forecast (QUEUE_FC_EVAL=_fc<k>), same LightGBM models.
Run: python -m tfb.queue.probs_avg --out results_top3/queue/final_v28fc/probs_avg.npz A/probs.npz B/probs.npz ...
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    zs = [np.load(f) for f in a.files]
    keys = sorted(zs[0].files)
    for z in zs[1:]:
        assert sorted(z.files) == keys
    out = {k: np.mean([z[k] for z in zs], 0).astype(np.float32) for k in keys}
    spread = np.mean([np.abs(zs[0][k] - out[k]).mean() for k in keys])
    np.savez(a.out, **out)
    print(f"{len(keys)} windows averaged over {len(zs)} runs (mean |run0 - avg| {spread:.4f}) -> {a.out}")


if __name__ == "__main__":
    main()
