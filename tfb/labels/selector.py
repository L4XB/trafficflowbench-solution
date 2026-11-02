"""Task-2 window selector replica: one greedy timeline, 360-minute spacing across both conditions.

Validity at origin T (history = T-60..T-5 observed, horizon = T+5..T+30 on the truth):
  onset:   no queued observation in the history, no truth queue at T+5..T+25, a truth queue at T+30
  ongoing: an established queue in the history (>=2 queued observations on one link), a truth
           queue somewhere in the horizon, persistence IoU (truth state at T-5 repeated) <= 0.9
  both:    history coverage (eligible share) >= 0.7
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfb.common import QUEUE_PANELS, STEPS, load_panel
from tfb.queue.data import HIST, HOR, SPACING, filled_speed, queued_observed, released_windows


def smooth3(sf: np.ndarray) -> np.ndarray:
    """3-step moving average along axis 1 (time of (days, 288, links)); the first and last step keep their values."""
    out = sf.copy()
    out[:, 1:-1] = (sf[:, :-2] + sf[:, 1:-1] + sf[:, 2:]) / 3
    return out


def validity(truth: np.ndarray, obs: np.ndarray, elig: np.ndarray) -> pd.DataFrame:
    """One row per (d, T) with the validity flags and diagnostics."""
    D, _, L = truth.shape
    rows = []
    Ts = np.arange(HIST, STEPS - HOR)
    for d in range(D):
        tr = truth[d]
        ob = obs[d]
        el = elig[d]
        # sliding windows via cumulative sums
        cs_ob = np.concatenate([np.zeros((1, L), int), np.cumsum(ob, 0)])
        cs_el = np.concatenate([np.zeros(1), np.cumsum(el.mean(1))])
        cs_tr = np.concatenate([np.zeros((1, L), int), np.cumsum(tr, 0)])
        for T in Ts:
            per_link = cs_ob[T] - cs_ob[T - HIST]
            cov = (cs_el[T] - cs_el[T - HIST]) / HIST
            hor_any_link = cs_tr[T + 1 + HOR] - cs_tr[T + 1]
            hor_n = int(hor_any_link.sum())
            hor_first5 = int((cs_tr[T + HOR] - cs_tr[T + 1]).sum())
            n30 = int(tr[T + HOR].sum())
            est = bool((per_link >= 2).any())
            anyq = bool(per_link.any())
            last = tr[T - 1]
            inter = int((last[None, :] & tr[T + 1 : T + 1 + HOR]).sum())
            union = int((last[None, :] | tr[T + 1 : T + 1 + HOR]).sum())
            piou = 1.0 if union == 0 else inter / union
            rows.append((d, T, cov, anyq, est, hor_n, hor_first5, n30, piou))
    v = pd.DataFrame(rows, columns=["d", "T", "cov", "hist_any", "established", "hor_n", "hor_first5", "n30", "piou"])
    v["onset_ok"] = (v["cov"] >= 0.7) & ~v.hist_any & (v.hor_first5 == 0) & (v.n30 > 0)
    v["ongoing_ok"] = (v["cov"] >= 0.7) & v.established & (v.hor_n > 0) & (v.piou <= 0.9)
    v["abs"] = v.d * STEPS + v["T"]
    return v


def merged_greedy(v: pd.DataFrame, quota: int | None = 5) -> pd.DataFrame:
    """Scan the timeline once; select a valid origin of either condition when spaced >= 360 min."""
    sel, last, n = [], -(10**9), {"onset": 0, "ongoing": 0}
    for r in v.itertuples():
        if r.abs - last < SPACING:
            continue
        cond = None
        if r.onset_ok and (quota is None or n["onset"] < quota):
            cond = "onset"
        elif r.ongoing_ok and (quota is None or n["ongoing"] < quota):
            cond = "ongoing"
        if cond is None:
            continue
        sel.append((r.d, r.T, cond, r.piou, r.n30, r.hor_n))
        last = r.abs
        n[cond] += 1
        if quota is not None and n["onset"] >= quota and n["ongoing"] >= quota:
            break
    return pd.DataFrame(sel, columns=["d", "T", "condition", "piou", "n30", "hor_n"])


if __name__ == "__main__":
    tot = {"exact": 0, "n": 0}
    for panel in QUEUE_PANELS:
        p = load_panel(panel, "train")
        sf = filled_speed(p)
        obs = queued_observed(p)
        elig = p["elig"]
        vcut = p.param("v_cut")[None, None, :]
        w = released_windows(panel, "train")
        rel = [(p.dates.index(r.date), int(r.T), r.condition[6:]) for r in w.itertuples()]
        rel = sorted(rel, key=lambda x: x[0] * STEPS + x[1])
        for name, truth in (("raw", sf <= vcut), ("smooth3", smooth3(sf) <= vcut)):
            v = validity(truth, obs, elig)
            g = merged_greedy(v, quota=5)
            got = [(r.d, r.T, r.condition) for r in g.itertuples()]
            exact = sum(1 for a, b in zip(got, rel) if a == b)
            near = sum(1 for a, b in zip(got, rel) if a[2] == b[2] and abs(a[0] * STEPS + a[1] - b[0] * STEPS - b[1]) <= 2)
            print(f"{panel:10s} {name:8s} exact {exact:2d}/10  within2steps {near:2d}/10 | got {got}")
            if name == "smooth3":
                tot["exact"] += exact
                tot["n"] += 10
        print(f"{'':10s} released      {rel}")
    print("smooth3 total exact", tot)
