"""Denoise the train speed field with the state U-Net: every cell is predicted with itself hidden.

Two complementary random 50 % masks per day; the prediction at a hidden cell is E[obs | context],
i.e. an estimate of the noise-free state the organizer thresholds for the queue truth.
Run: python -m tfb.labels.denoise [--panels ...]
Writes cache_top3/<panel>/train_speed_dn.npy (float32, all cells).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from tfb.common import CACHE, PANELS
from tfb.state.data import load_arrays
from tfb.state.predict import load_nets
from tfb.state.train import predict_full

CKPTS = ["results_top3/state/unet_s_v2/best.pt", "results_top3/state/unet_s_v4_seed43/best.pt", "results_top3/state/unet_s_v5_w05/best.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--ckpts", nargs="*", default=CKPTS)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--suffix", default="", help="output name suffix (default dn)")
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nets = load_nets(a.ckpts, dev)
    rng = np.random.default_rng(7)
    for pn in a.panels:
        arr = load_arrays(pn, "train", with_trans=False)
        D, T, L = arr.v.shape
        out = np.full((D, T, L), np.nan, np.float32)
        outq = np.full((D, T, L), np.nan, np.float32)
        cnt = np.zeros((D, T, L), np.int8)
        maskA = rng.random((D, T, L)) < 0.5
        for mask in (maskA, ~maskA):
            arr.target = mask  # full_panel_batch hides target cells from the input
            for i in range(0, D, a.chunk):
                days = np.arange(i, min(D, i + a.chunk))
                preds = [predict_full(net, arr, days, dev) for net in nets]
                pv = np.mean([pr[0] for pr in preds], 0)
                pq = np.mean([pr[1] for pr in preds], 0)
                sel = mask[days]
                tmp = out[days]
                tmp[sel] = pv[sel]
                out[days] = tmp
                cnt[days] += sel
                tmq = outq[days]
                tmq[sel] = pq[sel]
                outq[days] = tmq
        assert (cnt == 1).all(), "every cell must be predicted exactly once"
        np.save(CACHE / pn / (a.suffix and f"train_speed_{a.suffix}.npy" or "train_speed_dn.npy"), out)
        np.save(CACHE / pn / (a.suffix and f"train_flow_{a.suffix}.npy" or "train_flow_dn.npy"), outq)
        fin = np.isfinite(arr.v)
        rmse = float(np.sqrt(np.mean((out[fin] - arr.v[fin]) ** 2)))
        print(f"{pn}: denoised {D} days, rmse vs observed {rmse:.3f} km/h ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
