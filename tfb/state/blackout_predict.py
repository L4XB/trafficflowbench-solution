"""U-Net predictions with simulated evaluation-month blackouts on ALL train days (for the blackout corrector).

Same blackout placement as blackout_eval.py (greedy >= 360-min spaced released train windows; steps T+1..T+18 blank on every
link; the raw history T-12..T-1 observed, the origin T keeps its Task-1 mask). Writes results_top3/state/<out>/<panel>.npz with days, wins (n, 2) = (day, T),
pvB, pqB (float16, D x 288 x L).

Run: python -m tfb.state.blackout_predict --ckpts <...> --out bo_pred_long46 [--device cpu]
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch

from tfb.common import PROJECT, QUEUE_PANELS
from tfb.state.blackout_eval import BLACK, HIST, blackout_windows
from tfb.state.data import load_arrays
from tfb.state.predict import ensemble_predict, load_nets


def apply_blackouts(a, wins):
    """Copy of the panel arrays with the blackouts blank and the history rows un-targeted."""
    v, q, occ, tgt = a.v.copy(), a.q.copy(), a.occ.copy(), a.target.copy()
    for d, T in wins:
        v[d, T + 1 : T + 1 + BLACK] = np.nan
        q[d, T + 1 : T + 1 + BLACK] = np.nan
        occ[d, T + 1 : T + 1 + BLACK] = np.nan
        tgt[d, T - HIST : T] = False  # raw history T-12..T-1; the origin T keeps its Task-1 mask
    return dataclasses.replace(a, v=v, q=q, occ=occ, target=tgt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--panels", nargs="*", default=QUEUE_PANELS)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--placement-seed", type=int, default=None, help="random-order greedy placement (default: time order)")
    a_ = ap.parse_args()
    torch.set_num_threads(a_.threads)
    dev = torch.device(a_.device)
    nets = load_nets(a_.ckpts, dev)
    out = PROJECT / "results_top3" / "state" / a_.out
    out.mkdir(parents=True, exist_ok=True)
    for pn in a_.panels:
        a = load_arrays(pn, "train", with_trans=False)
        days = np.arange(a.n_days)
        wins = blackout_windows(pn, a.dates, days, a_.placement_seed)
        b = apply_blackouts(a, wins)
        pv, pq = ensemble_predict(nets, b, days, dev)
        np.savez_compressed(
            out / f"{pn}.npz", days=days, wins=np.array(wins, dtype=np.int32), pvB=pv.astype(np.float16), pqB=pq.astype(np.float16)
        )
        print(f"{pn}: {len(wins)} blackouts on {len(days)} days", flush=True)


if __name__ == "__main__":
    main()
