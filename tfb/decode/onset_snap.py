"""Onset: snap each decoded link cluster to its site's empirical onset block.

Onset truths are almost always one of a few fixed link blocks per panel (train months: D7_I405_S {32,33,35-39} 146x vs partial
variants <= 5x, D7_I10_E 4-10 222x, D12_I5_S {10,12,13,15} 66x, D7_I210_W {6,7} 79x vs {6} 5x; the March selector replica shows
the same blocks), because a breakdown queues the whole short-link stretch upstream of the bottleneck within one 5-minute step. The
per-link expected-IoU decode assumes independent links and keeps only the most likely links of a block when the site's activation is
uncertain. For each cluster (runs split at gaps > 2) of the decoded set, the blocks of the fit windows that contain the cluster's
highest-probability link are collected; if the most frequent one has >= --min-count occurrences and >= --min-share of them, the
cluster is replaced by that block (only blocks whose links all have template rows). Site selection is unchanged.
Options (off by default): --nearest-gap G replaces a cluster whose top link lies in no frequent block by the nearest frequent block
within G links; --zswap-lo / --zswap-hi replace a whole set whose live max (--live) is < lo by the frequent block (not overlapping
it) with the highest live max if that is >= hi.
Run: python -m tfb.decode.onset_snap --template pred_gate.csv --out pred_snap.csv
     python -m tfb.decode.onset_snap --forward   (fit Jun-Nov blocks, Dec-Feb windows, cached forward probabilities)
"""

from __future__ import annotations

import argparse
import collections
import zlib

import numpy as np
import pandas as pd

from tfb.common import PROJECT, QUEUE_PANELS, load_panel, t_index
from tfb.queue.data import HOR, released_windows
from tfb.queue.evaluation import windows_path

Q = PROJECT / "results_top3" / "queue"


def runs(ix) -> list:
    """Sorted link indices split into clusters at gaps > 2 (empty list for no links)."""
    ix = np.asarray(sorted(ix), int)
    return np.split(ix, np.where(np.diff(ix) > 2)[0] + 1) if len(ix) else []


def site_blocks(feat_file: str) -> dict:
    """panel -> Counter of truth blocks (tuples of link indices) over the onset windows of a feature cache."""
    w = pd.read_csv(windows_path("mix"))
    z = np.load(Q / feat_file, allow_pickle=True)
    k, y = z["keys"], z["y"].astype(bool)
    out = collections.defaultdict(collections.Counter)
    for widx in np.unique(k[:, 0]):
        m = k[:, 0] == widx
        for b in runs(k[m, 1][y[m]]):
            out[w.loc[widx, "panel"]][tuple(int(v) for v in b)] += 1
    return out


def load_blocks(src: str) -> dict:
    """Catalogue from a feature cache (site_blocks) or from its JSON export (tfb.decode.export_snap_blocks, same order and counts)."""
    if not src.endswith(".json"):
        return site_blocks(src)
    import json

    raw = json.loads((Q / src).read_text())
    out = collections.defaultdict(collections.Counter)
    for pn, items in raw.items():
        for b, c in items:
            out[pn][tuple(int(v) for v in b)] = int(c)
    return out


def snap(S: set, pp: np.ndarray, rows: set, blocks: collections.Counter, min_count: int, min_share: float) -> set:
    """Replace each cluster of S by the most frequent block that holds the cluster's top-pp link (blocks with all links in rows only).

    Applied when that block has >= min_count occurrences and >= min_share of the occurrences of all such blocks.
    """
    new = set(S)
    for c in runs(S):
        top = int(c[np.argmax(pp[c])])
        ov = [(b, f) for b, f in blocks.items() if top in b and all(l in rows for l in b)]
        if not ov:
            continue
        b, f = max(ov, key=lambda z: z[1])
        if f >= min_count and f >= min_share * sum(g for _, g in ov):
            new = (new - {int(v) for v in c}) | set(b)
    return new


def nearest(S: set, pp: np.ndarray, rows: set, blocks: collections.Counter, gap: int, min_count: int) -> set:
    """Replace clusters whose top-pp link is in no block with >= min_count by the nearest such block (links in rows) within `gap` links."""
    new = set(S)
    for c in runs(S):
        top = int(c[np.argmax(pp[c])])
        if any(top in b for b, f in blocks.items() if f >= min_count):
            continue
        near = [(min(abs(l - v) for l in b for v in c), -f, b) for b, f in blocks.items() if f >= min_count and all(l in rows for l in b)]
        near = [z for z in near if z[0] <= gap]
        if near:
            new = (new - {int(v) for v in c}) | set(min(near)[2])
    return new


def zswap(S: set, pl: np.ndarray, rows: set, blocks: collections.Counter, lo: float, hi: float, min_count: int) -> set:
    """S unless its live max (pl) is < lo; then the frequent block disjoint from S with the highest live max, if that max is >= hi."""
    if not S or pl[sorted(S)].max() >= lo:
        return set(S)
    other = [(float(pl[list(b)].max()), f, b) for b, f in blocks.items() if f >= min_count and all(l in rows for l in b) and not set(b) & S]
    if not other:
        return set(S)
    lm, _, b = max(other)
    return set(b) if lm >= hi else set(S)


def forward(a) -> None:
    """Forward check: gate (ratio 0.7) and snap the Dec-Feb onset decodes with the Jun-Nov blocks; print the IoU change."""
    from tfb.queue import train as queue_train
    from tfb.queue.train import decode

    tag = "2030-12-2031-01-2031-02_v6_mix_fillcausal_nodrop"
    blocks = site_blocks(f"feat_onset_fit_{tag}.npz")
    w = pd.read_csv(windows_path("mix"))
    zt = np.load(Q / f"feat_onset_test_{tag}.npz", allow_pickle=True)
    kt, yt = zt["keys"], zt["y"].astype(bool)
    c = np.load(Q / "onset_decode_fwd_probs.npz")
    prod, live = c["prod"], c["live"]
    rec = []
    for widx in np.unique(kt[:, 0]):
        m = kt[:, 0] == widx
        o = np.argsort(kt[m, 1])
        links = kt[m, 1][o]
        pp = np.zeros(links.max() + 1)
        pp[links] = prod[m][o]
        pl = np.zeros_like(pp)
        pl[links] = live[m][o]
        t = {int(v) for v in links[yt[m][o]]}
        queue_train.RNG = np.random.default_rng(zlib.crc32(str(widx).encode()))
        S = {int(v) for v in np.where(decode(pp, "eiou"))[0]}
        cl = runs(S)
        if len(cl) >= 2:  # live cluster gate, ratio 0.7
            best = max(pl[x].max() for x in cl)
            S = {int(v) for x in cl if pl[x].max() >= 0.7 * best for v in x}
        iou = lambda s, t=t: len(s & t) / len(s | t) if (s | t) else 1.0
        S2 = snap(S, pp, {int(v) for v in links}, blocks[w.loc[widx, "panel"]], a.min_count, a.min_share)
        rec.append({"panel": w.loc[widx, "panel"], "base": iou(S), "snap": iou(S2), "ch": S2 != S})
    r = pd.DataFrame(rec)
    d = r.snap - r.base
    print(
        f"forward {len(r)} onset windows (gate 0.7): base {r.base.mean():.4f}, snap {r.snap.mean():.4f} ({d.mean():+.4f} se {d.std() / np.sqrt(len(d)):.4f}); "
        f"changed {int(r.ch.sum())}, on them {d[r.ch].mean():+.4f}; per panel "
        + str({p: round(float(g.mean()), 4) for p, g in d.groupby(r.panel)})
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template")
    ap.add_argument("--out")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--prod", default="final_v27mixcausal/probs.npz")
    ap.add_argument("--blocks", default="feat_onset_fit_all_v6_mix_fillcausal_nodrop.npz")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-share", type=float, default=0.75)
    ap.add_argument("--splits", nargs="*", default=["validation", "private"])
    ap.add_argument("--nearest-gap", type=int, default=0)
    ap.add_argument("--live", default="final_v27live/probs.npz")
    ap.add_argument("--zswap-lo", type=float, default=-1.0)
    ap.add_argument("--zswap-hi", type=float, default=2.0)
    a = ap.parse_args()
    if a.forward:
        forward(a)
        return
    blocks = load_blocks(a.blocks)
    zp = np.load(Q / a.prod, allow_pickle=True)
    pred = pd.read_csv(Q / a.template)
    changed = {}
    zl = np.load(Q / a.live, allow_pickle=True) if a.zswap_lo > 0 else None
    for split in a.splits:
        changed[split] = 0
        for pn in QUEUE_PANELS:
            p = load_panel(pn, split)
            li = {str(l): i for i, l in enumerate(p.links)}
            r = released_windows(pn, split)
            for x in r[r.condition == "queue_onset"].itertuples():
                sel = pred.window_id == x.window_id
                idx = pred.index[sel]
                ts = t_index(pred.loc[sel, "timestamp"].to_numpy())
                lk = np.array([li[str(l)] for l in pred.loc[sel, "link_id"]])
                c = int(x.T) + HOR
                on = (pred.loc[sel, "queue_pred"].to_numpy() == 1) & (ts == c)
                S = {int(v) for v in lk[on]}
                if not S:
                    continue
                rows = {int(v) for v in lk[ts == c]}
                pp = np.asarray(zp[x.window_id], float).ravel()
                S2 = snap(S, pp, rows, blocks[pn], a.min_count, a.min_share)
                if a.nearest_gap > 0:
                    S2 = nearest(S2, pp, rows, blocks[pn], a.nearest_gap, a.min_count)
                if zl is not None:
                    S2 = zswap(S2, np.asarray(zl[x.window_id], float).ravel(), rows, blocks[pn], a.zswap_lo, a.zswap_hi, a.min_count)
                if S2 != S:
                    print(f"  snap {x.window_id}: {sorted(S)} -> {sorted(S2)}")
                    new = np.where(ts == c, np.isin(lk, sorted(S2)), False).astype(int)
                    changed[split] += int((pred.loc[idx, "queue_pred"].to_numpy() != new).sum())
                    pred.loc[idx, "queue_pred"] = new
    pred.to_csv(Q / a.out, index=False)
    print(f"onset cells changed {changed} -> {a.out}")


if __name__ == "__main__":
    main()
