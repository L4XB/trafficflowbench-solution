"""Causal U-Net fills for Task-2 window histories (legal under the organizer's <= T rule).

For every window (d, T) the state ensemble predicts day d with every observation, ramp count and quality
field AFTER T hidden. Only rows [T-HIST, T] are kept: observed cells keep their value, gaps take the fill.
Train windows use the unmasked layer for the history rows [T-HIST, T-1] and the masked layer elsewhere
(rows < T-HIST and the origin T), which mimics what the evaluation splits publish.

Writes cache_top3/<panel>/<split>_causalfill.npz with d, T (window index order), v, q (n, HIST+1, L).
Run: python -m tfb.queue.fill_causal --split train|validation|private [--device cpu]
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import pandas as pd
import torch

from tfb.common import CACHE, PROJECT, QUEUE_PANELS, load_panel
from tfb.queue.data import HIST, released_windows
from tfb.state.data import load_arrays
from tfb.state.predict import load_nets
from tfb.state.train import predict_full

CKPTS = [
    "results_top3/state/unet_h31_long46/best.pt",
    "results_top3/state/unet_h31_long48/best.pt",
    "results_top3/state/unet_h31_long49/best.pt",
]


def window_table(pn: str, split: str, truth: str) -> pd.DataFrame:
    """Window origins (d, T) of a panel: windows_train_<truth>.csv for train, else the released windows (+ REPLICA_WINDOWS if set)."""
    if split == "train":
        w = pd.read_csv(PROJECT / "results_top3" / "queue" / f"windows_train_{truth}.csv")
        w = w[w.panel == pn][["d", "T"]].drop_duplicates().reset_index(drop=True)
    else:
        p = load_panel(pn, split)
        r = released_windows(pn, split)
        w = pd.DataFrame({"d": [p.dates.index(x) for x in r.date], "T": r["T"].astype(int).to_numpy()})
        import os

        if os.environ.get("REPLICA_WINDOWS"):  # add the selector-replica windows of the split (offline test bed)
            rw = pd.read_csv(os.environ["REPLICA_WINDOWS"])
            rw = rw[rw.panel == pn][["d", "T"]]
            w = pd.concat([w, rw]).drop_duplicates().reset_index(drop=True)
    return w


def causal_inputs(arr_in, arr_hist, wins: pd.DataFrame, split: str):
    """Stack one synthetic day per window with everything after T hidden."""
    sel = wins.d.to_numpy()
    v = arr_in.v[sel].copy()
    q = arr_in.q[sel].copy()
    occ = arr_in.occ[sel].copy()
    pct = arr_in.pct[sel].copy()
    elig = arr_in.elig[sel].copy()
    ron = arr_in.ramp_on[sel].copy()
    roff = arr_in.ramp_off[sel].copy()
    ron_ok = arr_in.ramp_on_ok[sel].copy()
    roff_ok = arr_in.ramp_off_ok[sel].copy()
    for i, (d, T) in enumerate(zip(wins.d.to_numpy(), wins["T"].to_numpy())):
        if split == "train":  # history rows from the unmasked layer (natural gaps only)
            v[i, T - HIST : T] = arr_hist.v[d, T - HIST : T]
            q[i, T - HIST : T] = arr_hist.q[d, T - HIST : T]
            occ[i, T - HIST : T] = arr_hist.occ[d, T - HIST : T]
        v[i, T + 1 :] = np.nan
        q[i, T + 1 :] = np.nan
        occ[i, T + 1 :] = np.nan
        pct[i, T + 1 :] = 0
        elig[i, T + 1 :] = False
        ron[i, T + 1 :] = 0.0
        roff[i, T + 1 :] = 0.0
        ron_ok[i, T + 1 :] = False
        roff_ok[i, T + 1 :] = False
    return dataclasses.replace(
        arr_in,
        dates=[arr_in.dates[d] for d in sel],
        regimes=np.asarray(arr_in.regimes)[sel],
        v=v,
        q=q,
        occ=occ,
        pct=pct,
        elig=elig,
        target=np.zeros(v.shape, bool),
        ramp_on=ron,
        ramp_off=roff,
        ramp_on_ok=ron_ok,
        ramp_off_ok=roff_ok,
        weekday=arr_in.weekday[sel],
        trans_ok=None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--panels", nargs="*", default=QUEUE_PANELS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--truth", default="mix")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ckpts", nargs="*", default=CKPTS)
    ap.add_argument("--merge", action="store_true", help="add missing windows to an existing file")
    ap.add_argument("--suffix", default="", help="output file suffix")
    ap.add_argument(
        "--prefix",
        action="store_true",
        help="evaluation splits: also save the whole causal prefix rows 0..T of each window "
        "(observed cells kept, gaps filled) to <split>_causalprefix<suffix>.npz -- the same-day context before the history must "
        "not come from the whole-day reconstruction, which has seen the post-buffer observations",
    )
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(8)
    dev = torch.device(a.device if a.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    nets = load_nets(a.ckpts, dev)
    for pn in a.panels:
        wins = window_table(pn, a.split, a.truth)
        outf = CACHE / pn / f"{a.split}_causalfill{a.suffix}.npz"
        prev = None
        if a.merge and outf.exists():  # only compute windows that are not cached yet, then merge
            prev = dict(np.load(outf))
            have = set(zip(prev["d"].tolist(), prev["T"].tolist()))
            wins = wins[[(int(d), int(t)) not in have for d, t in zip(wins.d, wins["T"])]].reset_index(drop=True)
            if len(wins) == 0:
                print(f"{pn} {a.split}: nothing new", flush=True)
                continue
        if a.split == "train":
            arr_hist = load_arrays(pn, "train", with_trans=False)  # unmasked layer
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
        hq = np.zeros_like(hv)  # causal U-Net forecast for T+5..T+30
        if a.prefix:
            prv = np.full((len(wins), 288, L), np.nan, np.float32)
            prq = np.full_like(prv, np.nan)
        for s in range(0, len(wins), a.batch):
            wb = wins.iloc[s : s + a.batch]
            ab = causal_inputs(arr_in, arr_hist, wb, a.split)
            days = np.arange(len(wb))
            preds = [predict_full(net, ab, days, dev) for net in nets]
            pv = np.mean([x[0] for x in preds], 0)
            pq = np.mean([x[1] for x in preds], 0)
            for i, T in enumerate(wb["T"].to_numpy()):
                ov = ab.v[i, T - HIST : T + 1]
                oq = ab.q[i, T - HIST : T + 1]
                outv[s + i] = np.where(np.isfinite(ov), ov, pv[i, T - HIST : T + 1])
                outq[s + i] = np.where(np.isfinite(oq), oq, pq[i, T - HIST : T + 1])
                hv[s + i] = pv[i, T + 1 : T + 7]
                hq[s + i] = pq[i, T + 1 : T + 7]
                if a.prefix:
                    av = ab.v[i, : T + 1]
                    aq = ab.q[i, : T + 1]
                    prv[s + i, : T + 1] = np.where(np.isfinite(av), av, pv[i, : T + 1])
                    prq[s + i, : T + 1] = np.where(np.isfinite(aq), aq, pq[i, : T + 1])
        dd, tt = wins.d.to_numpy(), wins["T"].to_numpy()
        if prev is not None:
            dd = np.concatenate([prev["d"], dd])
            tt = np.concatenate([prev["T"], tt])
            outv = np.concatenate([prev["v"], outv])
            outq = np.concatenate([prev["q"], outq])
            if "hv" in prev:
                hv = np.concatenate([prev["hv"], hv])
                hq = np.concatenate([prev["hq"], hq])
        np.savez(outf, d=dd, T=tt, v=outv, q=outq, hv=hv, hq=hq)
        if a.prefix:
            assert prev is None, "--prefix does not support --merge"
            np.savez(CACHE / pn / f"{a.split}_causalprefix{a.suffix}.npz", d=wins.d.to_numpy(), T=wins["T"].to_numpy(), v=prv, q=prq)
        print(f"{pn} {a.split}: {len(wins)} windows causally filled ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
