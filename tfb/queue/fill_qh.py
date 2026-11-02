"""Causal horizon forecasts from a queue-head forecaster (tfb/state/train_qh.py): speed, flow AND queue probability for T+5..T+30.

Same windows and causal inputs as tfb/queue/fill_causal.py (everything after T hidden; train windows use the unmasked history rows).
Writes cache_top3/<panel>/<split>_causalfill<suffix>.npz with d, T, hv, hq, hp (n, 6, L) and v, q (n, HIST+1, L) as in the fill files.
Run: python -m tfb.queue.fill_qh --split train --suffix _fc0qh --ckpts results_top3/state/fc_fold0qh/best.pt --device mps
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import torch

from tfb.common import CACHE, QUEUE_PANELS, load_panel
from tfb.queue.data import HIST
from tfb.queue.fill_causal import causal_inputs, window_table
from tfb.state.data import N_FEATURES, load_arrays
from tfb.state.model import build_model
from tfb.state.train_qh import predict_full_qh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--panels", nargs="*", default=QUEUE_PANELS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--truth", default="mix")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--suffix", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(8)
    dev = torch.device(a.device)
    nets = []
    for c in a.ckpts:
        ck = torch.load(c, map_location=dev)
        assert int(ck["mcfg"].get("out_ch", 2)) == 3, f"{c} has no queue head"
        net = build_model(ck["mcfg"], N_FEATURES).to(dev)
        net.load_state_dict(ck["ema"])
        net.eval()
        nets.append(net)
    for pn in a.panels:
        wins = window_table(pn, a.split, a.truth)
        outf = CACHE / pn / f"{a.split}_causalfill{a.suffix}.npz"
        if a.split == "train":
            arr_hist = load_arrays(pn, "train", with_trans=False)
            p = load_panel(pn, "train")
            arr_in = dataclasses.replace(
                arr_hist,
                v=np.asarray(p["speed_m"], np.float32),
                q=np.asarray(p["flow_m"], np.float32),
                occ=np.asarray(p["occ_m"], np.float32),
            )
        else:
            arr_in = load_arrays(pn, a.split, with_trans=False)
            arr_hist = arr_in
        L = arr_in.v.shape[2]
        outv = np.zeros((len(wins), HIST + 1, L), np.float32)
        outq = np.zeros_like(outv)
        hv = np.zeros((len(wins), 6, L), np.float32)
        hq = np.zeros_like(hv)
        hp = np.zeros_like(hv)
        for s in range(0, len(wins), a.batch):
            wb = wins.iloc[s : s + a.batch]
            ab = causal_inputs(arr_in, arr_hist, wb, a.split)
            days = np.arange(len(wb))
            preds = [predict_full_qh(net, ab, days, dev) for net in nets]
            pv = np.mean([x[0] for x in preds], 0)
            pq = np.mean([x[1] for x in preds], 0)
            pp = np.mean([x[2] for x in preds], 0)
            for i, T in enumerate(wb["T"].to_numpy()):
                ov = ab.v[i, T - HIST : T + 1]
                oq = ab.q[i, T - HIST : T + 1]
                outv[s + i] = np.where(np.isfinite(ov), ov, pv[i, T - HIST : T + 1])
                outq[s + i] = np.where(np.isfinite(oq), oq, pq[i, T - HIST : T + 1])
                hv[s + i] = pv[i, T + 1 : T + 7]
                hq[s + i] = pq[i, T + 1 : T + 7]
                hp[s + i] = pp[i, T + 1 : T + 7]
        np.savez(outf, d=wins.d.to_numpy(), T=wins["T"].to_numpy(), v=outv, q=outq, hv=hv, hq=hq, hp=hp)
        print(f"{pn} {a.split}: {len(wins)} windows ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
