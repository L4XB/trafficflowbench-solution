"""Replace the windows of a probability dump by those of another dump (same shapes); all other windows are kept.

Run: python -m tfb.queue.probs_override --base probs_blend10pp.npz --override probs_priv.npz --out probs_blend10pp_B.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--override", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = np.load(a.base)
    over = np.load(a.override)
    out = {k: base[k] for k in base.files}
    for k in over.files:
        assert k in out and out[k].shape == over[k].shape, k
        out[k] = over[k]
    np.savez(a.out, **out)
    print(f"{len(over.files)} of {len(out)} windows replaced -> {a.out}")


if __name__ == "__main__":
    main()
