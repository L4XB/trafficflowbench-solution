"""Ongoing decode with a per-window threshold from the eval month's congestion level (legal: data before the window's origin).

ratio = queued share of the panel (visible cells, 07-19 h) on the split's days before d plus the window's own day before T, over the
train months' share for the same hours; thr = clip(base x ratio^-gamma, lo, hi). The finals use gamma = -0.25, so a quieter month than
train LOWERS the threshold: the March board preferred this sign to the queueing intuition (gamma > 0, a quieter month raises it; see
the report's calibration section). The train months carry no panel-level shift, so gamma was set on the validation month.
Run: python -m tfb.decode.ongoing_ratio_thr --probs probs_blend5nc.npz --template pred_v47og40_cgr07.csv --gamma 0.5 --out X.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tfb.common import PROJECT, QUEUE_PANELS, load_panel, t_index
from tfb.queue.data import released_windows

Q = PROJECT / "results_top3" / "queue"


def ratios(split: str, mode: str = "both") -> dict:
    """window_id -> queued share of the panel's visible 07-19 h cells up to the origin over its train share, per released ongoing window.

    mode: "both" pools the split's earlier days and the window's own day up to T, "prevdays" / "sameday" use one of them; no cells -> 1.0.
    """
    out = {}
    for pn in QUEUE_PANELS:
        tr = load_panel(pn, "train")
        vc = tr.param("v_cut")
        s = np.asarray(tr["speed_m"], np.float32)[:, 84:228]
        f = np.isfinite(s)
        base = ((s <= vc[None, None, :]) & f).sum() / f.sum()
        p = load_panel(pn, split)
        sp = np.asarray(p["speed_m"], np.float32)
        r = released_windows(pn, split)
        for x in r[r.condition == "queue_ongoing"].itertuples():
            d = p.dates.index(x.date)
            T = int(x.T)
            prev = sp[:d, 84:228]
            same = sp[d, 84 : min(228, T + 1)]
            qp = ((prev <= vc[None, None, :]) & np.isfinite(prev)).sum()
            qs = ((same <= vc[None, :]) & np.isfinite(same)).sum()
            np_, ns = np.isfinite(prev).sum(), np.isfinite(same).sum()
            q, n = {"both": (qp + qs, np_ + ns), "sameday": (qs, ns), "prevdays": (qp, np_)}[mode]
            out[x.window_id] = (q / n) / base if n > 0 else 1.0  # no data -> neutral
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--base", type=float, default=0.40)
    ap.add_argument("--lo", type=float, default=0.30)
    ap.add_argument("--hi", type=float, default=0.55)
    ap.add_argument("--ratio-mode", default="both", choices=["both", "sameday", "prevdays"])
    ap.add_argument("--quiet-only", action="store_true", help="adjust only windows with ratio < 1 (busy panels keep the base threshold)")
    a = ap.parse_args()
    z = np.load(Q / a.probs, allow_pickle=True)
    pred = pd.read_csv(Q / a.template)
    changed = {}
    for split in ("validation", "private"):
        rr = ratios(split, a.ratio_mode)
        changed[split] = 0
        for pn in QUEUE_PANELS:
            p = load_panel(pn, split)
            li = {str(l): i for i, l in enumerate(p.links)}
            for wid, ratio in rr.items():
                if f"_{pn}_" not in wid:
                    continue
                thr = float(np.clip(a.base * max(ratio, 1e-3) ** (-a.gamma), a.lo, a.hi)) if (ratio < 1 or not a.quiet_only) else a.base
                prob = np.asarray(z[wid], float)  # (HOR, L)
                sel = pred.window_id == wid
                idx = pred.index[sel]
                T = int(released_windows(pn, split).set_index("window_id").loc[wid, "T"])
                h = t_index(pred.loc[sel, "timestamp"].to_numpy()) - T - 1
                lk = np.array([li[str(l)] for l in pred.loc[sel, "link_id"]])
                new = (prob[h, lk] >= thr).astype(int)
                changed[split] += int((pred.loc[idx, "queue_pred"].to_numpy() != new).sum())
                pred.loc[idx, "queue_pred"] = new
    pred.to_csv(Q / a.out, index=False)
    print(f"gamma {a.gamma}: ongoing cells changed {changed} -> {a.out}")


if __name__ == "__main__":
    main()
