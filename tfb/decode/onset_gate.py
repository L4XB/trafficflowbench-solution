"""Onset: drop predicted clusters the live-only model sees no precursors for.

The production onset sets often hold two clusters (e.g. an old bottleneck and a new one). In the evaluation months some old
bottlenecks are dormant; a live-only model (no climatology, no link ids, no static link attributes: final_v27live, QUEUE_ROBUST=3)
judges activation from the current state alone. Clusters (link runs separated by > 1 link) whose maximum live probability is below
--tau are removed, keeping at least the cluster with the highest live probability.
Run: python -m tfb.decode.onset_gate --template pred_v47_og40.csv --live final_v27live/probs.npz --tau 0.1 --out X.csv
Options (off by default): --redecode adds the expected-IoU decode of the production probabilities restricted to the candidate regions
(p >= --cand, +-1 link) that overlap the kept clusters of gated windows; --swap-lo/--swap-hi replace a set whose live max is < swap-lo by the unselected production candidate
cluster (p >= --cand, runs split at gaps > 2) whose live max is >= swap-hi.
"""

from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

from tfb.common import PROJECT, QUEUE_PANELS, load_panel, t_index
from tfb.queue import train as queue_train
from tfb.queue.data import HOR, released_windows
from tfb.queue.features import inmonth_day_feats
from tfb.queue.train import decode

Q = PROJECT / "results_top3" / "queue"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["validation", "private"])
    ap.add_argument(
        "--mode",
        default="tau",
        choices=["tau", "top", "ratio"],
        help="tau: keep clusters with live max >= tau; top: keep only the "
        "cluster with the highest live max; ratio: keep clusters with live max >= tau x the best cluster's",
    )
    ap.add_argument("--prod", default="final_v27mixcausal/probs.npz")
    ap.add_argument("--redecode", action="store_true")
    ap.add_argument("--swap-lo", type=float, default=-1.0)
    ap.add_argument("--swap-hi", type=float, default=2.0)
    ap.add_argument("--cand", type=float, default=0.12)
    ap.add_argument(
        "--dormant",
        type=float,
        default=-1.0,
        help="before the live gate drop clusters whose links never queued on the month's earlier days (whole-day share 0, tfb.queue.features.inmonth_day_feats) when another cluster's share is >= this value; run with the production environment",
    )
    a = ap.parse_args()
    z = np.load(Q / a.live, allow_pickle=True)
    pred = pd.read_csv(Q / a.template)
    changed = {s: 0 for s in a.splits}
    zp = np.load(Q / a.prod, allow_pickle=True) if (a.redecode or a.swap_lo > 0) else None
    pqs = {}

    def redecode(pp: np.ndarray, spans: list, wid: str) -> np.ndarray:
        """Expected-IoU decode of pp restricted to the spans (+-1 link), with the decode RNG seeded from the window id's CRC32."""
        q = np.zeros_like(pp)
        for c in spans:
            q[max(0, c[0] - 1) : c[-1] + 2] = pp[max(0, c[0] - 1) : c[-1] + 2]
        queue_train.RNG = np.random.default_rng(zlib.crc32(wid.encode()))
        return decode(q, "eiou")

    for pn in QUEUE_PANELS:
        for split in a.splits:
            p = load_panel(pn, split)
            li = {str(l): i for i, l in enumerate(p.links)}
            r = released_windows(pn, split)
            for x in r[r.condition == "queue_onset"].itertuples():
                c = int(x.T) + HOR
                pl = np.asarray(z[x.window_id], float).ravel()
                sel = pred.window_id == x.window_id
                idx = pred.index[sel]
                ts = t_index(pred.loc[sel, "timestamp"].to_numpy())
                lk = np.array([li[str(l)] for l in pred.loc[sel, "link_id"]])
                on = (pred.loc[sel, "queue_pred"].to_numpy() == 1) & (ts == c)
                S = np.zeros(len(li), bool)
                S[lk[on]] = True
                idxs = np.where(S)[0]
                if len(idxs) == 0:
                    continue
                if a.swap_lo > 0 and pl[idxs].max() < a.swap_lo:
                    pp = np.asarray(zp[x.window_id], float).ravel()
                    cand = np.where(pp >= a.cand)[0]
                    other = [
                        cc
                        for cc in (np.split(cand, np.where(np.diff(cand) > 2)[0] + 1) if len(cand) else [])
                        if not set(cc.tolist()) & set(idxs.tolist())
                    ]
                    other = [cc for cc in other if pl[cc].max() >= a.swap_hi]
                    if other:
                        best = max(other, key=lambda cc: pl[cc].max())
                        S2 = redecode(pp, [best], x.window_id)
                        new = np.where(ts == c, S2[lk], False).astype(int)
                        print(f"  swap {x.window_id}: {idxs.tolist()} -> {np.where(S2)[0].tolist()}")
                        changed[split] += int((pred.loc[idx, "queue_pred"].to_numpy() != new).sum())
                        pred.loc[idx, "queue_pred"] = new
                        continue
                cl = np.split(idxs, np.where(np.diff(idxs) > 2)[0] + 1)
                if len(cl) < 2:
                    continue
                n_in = len(cl)
                if a.dormant >= 0:
                    if (pn, split) not in pqs:
                        from tfb.queue.evaluation import PanelQ

                        pqs[(pn, split)] = PanelQ(pn, split, truth="mix")
                    pq = pqs[(pn, split)]
                    act = inmonth_day_feats(pq, pq.dates.index(x.date), int(x.T))[0][:, 0]
                    ac = [np.nanmax(act[cc]) for cc in cl]
                    if all(np.isfinite(ac)) and max(ac) >= a.dormant and min(ac) == 0:
                        print(
                            f"  dormant {x.window_id}: drop {[cc.tolist() for cc, v in zip(cl, ac) if v == 0]} (shares {np.round(ac, 2).tolist()})"
                        )
                        cl = [cc for cc, v in zip(cl, ac) if v > 0]
                best = max(cl, key=lambda cc: pl[cc].max())
                if a.mode == "top":
                    keep = [best]
                elif a.mode == "ratio":
                    keep = [cc for cc in cl if pl[cc].max() >= a.tau * pl[best].max()]
                else:
                    keep = [cc for cc in cl if pl[cc].max() >= a.tau] or [best]
                S2 = np.zeros_like(S)
                for cc in keep:
                    S2[cc] = True
                if a.redecode and len(keep) < n_in:
                    pp = np.asarray(zp[x.window_id], float).ravel()
                    cand = np.where(pp >= a.cand)[0]
                    regions = [
                        cc
                        for cc in (np.split(cand, np.where(np.diff(cand) > 2)[0] + 1) if len(cand) else [])
                        if any(set(cc.tolist()) & set(k.tolist()) for k in keep)
                    ] or keep
                    S3 = S2 | redecode(pp, regions, x.window_id)  # only adds links of the kept candidate regions
                    if (S3 != S2).any():
                        print(f"  redecode {x.window_id}: {np.where(S2)[0].tolist()} -> {np.where(S3)[0].tolist()}")
                        S2 = S3
                new = np.where(ts == c, S2[lk], False).astype(int)
                changed[split] += int((pred.loc[idx, "queue_pred"].to_numpy() != new).sum())
                pred.loc[idx, "queue_pred"] = new
    pred.to_csv(Q / a.out, index=False)
    print(f"tau {a.tau}: onset cells removed {changed} -> {a.out}")


if __name__ == "__main__":
    main()
