"""Ongoing: scale the per-window threshold (quiet-panel rule, as in tfb.decode.ongoing_ratio_thr) by the window's selection kind.

Kinds from data allowed at T (host answers: released origins <= T and the documented selection rule): "bound" = the window starts exactly
360 min after the previous selected window of the panel; "young" = the released history shows no link with >= 2 queued observations in
T-12..T-2 (the queue was not yet established at T-1). Others are change points.
Run: python -m tfb.decode.ongoing_kind_thr --template pred.csv --kind changepoint --scale 1.1 --out pred_out.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tfb.common import PROJECT, QUEUE_PANELS, RELEASE, STEPS, load_panel, t_index
from tfb.decode.ongoing_ratio_thr import ratios
from tfb.queue.data import released_windows

Q = PROJECT / "results_top3" / "queue"


def kinds(split: str) -> dict:
    """window_id -> "bound", "young" or "changepoint" (see the module docstring) for every released ongoing window of the split."""
    out = {}
    for pn in QUEUE_PANELS:
        p = load_panel(pn, split)
        vc = p.param("v_cut")
        li = {str(l): i for i, l in enumerate(p.links)}
        h = pd.read_parquet(RELEASE / "task2" / pn / split / "window_history.parquet")
        r = released_windows(pn, split).copy()
        r["abs"] = [p.dates.index(x.date) * STEPS + int(x.T) for x in r.itertuples()]
        r = r.sort_values("abs")
        r["gap"] = r["abs"].diff()
        for x in r[r.condition == "queue_ongoing"].itertuples():
            g = h[h.window_id == x.window_id]
            T = int(x.T)
            ti = t_index(g.timestamp.to_numpy()) - T
            lk = np.array([li[str(l)] for l in g.link_id])
            ok = (ti >= -12) & (ti <= -2) & g.speed_kmh.notna().to_numpy() & (g.is_score_eligible.to_numpy() == 1)
            Qm = np.zeros((11, len(li)), bool)
            Qm[ti[ok] + 12, lk[ok]] = g.speed_kmh.to_numpy()[ok] <= vc[lk[ok]]
            out[x.window_id] = "bound" if x.gap == 72 else ("young" if not (Qm.sum(0) >= 2).any() else "changepoint")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probs", default="probs_blend10pp.npz")
    ap.add_argument("--kind", required=True, choices=["young", "bound", "changepoint"])
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=-0.25)
    ap.add_argument("--base", type=float, default=0.38)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.55)
    a = ap.parse_args()
    z = np.load(Q / a.probs, allow_pickle=True)
    pred = pd.read_csv(Q / a.template)
    changed = {}
    for split in ("validation", "private"):
        kd = kinds(split)
        rr = ratios(split)
        changed[split] = 0
        for pn in QUEUE_PANELS:
            p = load_panel(pn, split)
            li = {str(l): i for i, l in enumerate(p.links)}
            r = released_windows(pn, split).set_index("window_id")
            for wid, kind in kd.items():
                if f"_{pn}_" not in wid or kind != a.kind:
                    continue
                thr = float(np.clip(a.base * max(rr[wid], 1e-3) ** (-a.gamma), a.lo, a.hi)) * a.scale
                sel = pred.window_id == wid
                idx = pred.index[sel]
                T = int(r.loc[wid, "T"])
                h = t_index(pred.loc[sel, "timestamp"].to_numpy()) - T - 1
                lk = np.array([li[str(l)] for l in pred.loc[sel, "link_id"]])
                new = (np.asarray(z[wid], float)[h, lk] >= thr).astype(int)
                changed[split] += int((pred.loc[idx, "queue_pred"].to_numpy() != new).sum())
                pred.loc[idx, "queue_pred"] = new
    pred.to_csv(Q / a.out, index=False)
    print(f"{a.kind} x{a.scale}: ongoing cells changed {changed} -> {a.out}")


if __name__ == "__main__":
    main()
