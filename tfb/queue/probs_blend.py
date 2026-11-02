"""Weighted blend of per-window probability dumps (keys present in all inputs): --weights w1 w2 ... files.
Run: python -m tfb.queue.probs_blend --out blend.npz --weights 0.5 0.5 a/probs.npz b/probs_avg.npz [--condition ongoing]
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ndim", type=int, default=2, help="2 = ongoing windows (HOR, L), 1 = onset (L,)")
    a = ap.parse_args()
    assert len(a.weights) == len(a.files)
    zs = [np.load(f) for f in a.files]
    keys = sorted(set.intersection(*[{k for k in z.files if z[k].ndim == a.ndim} for z in zs]))
    w = np.array(a.weights) / np.sum(a.weights)
    out = {k: sum(wi * z[k] for wi, z in zip(w, zs)).astype(np.float32) for k in keys}
    np.savez(a.out, **out)
    print(f"{len(keys)} windows blended with weights {np.round(w, 3).tolist()} -> {a.out}")


if __name__ == "__main__":
    main()
