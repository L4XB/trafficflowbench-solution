"""Re-decode the ONSET windows of a prediction table from a probability dump (thr or eiou); other rows unchanged.

Run: python -m tfb.decode.onset_redecode --probs P.npz --template pred.csv --mode eiou --out out.csv
"""

from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

from tfb.common import QUEUE_PANELS, load_panel, t_index
from tfb.decode.merge import window_conditions
from tfb.queue import train as queue_train
from tfb.queue.data import released_windows
from tfb.queue.train import decode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="thr")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the probabilities (clipped to 1) before decoding")
    ap.add_argument("--min-mass", type=float, default=0.0, help="rescale windows whose total probability is below this value up to it")
    a = ap.parse_args()
    cond = window_conditions()
    z = np.load(a.probs, allow_pickle=True)
    pred = pd.read_csv(a.template)
    meta = {}
    for pn in QUEUE_PANELS:
        for split in ("validation", "private"):
            w = released_windows(pn, split)
            links = {l: i for i, l in enumerate(load_panel(pn, split).links)}
            for r in w.itertuples():
                meta[r.window_id] = (int(r.T), links)
    sel = pred.window_id.map(cond) == "queue_onset"
    ts = t_index(pred.loc[sel, "timestamp"].to_numpy())
    wids = pred.loc[sel, "window_id"].to_numpy()
    lids = pred.loc[sel, "link_id"].to_numpy()
    new = np.zeros(int(sel.sum()), int)
    cache = {}
    old_vals = pred.loc[sel, "queue_pred"].to_numpy()
    keep_tpl = set()
    if a.min_mass > 0:  # only the low-mass windows are re-decoded; the rest keep the template
        keep_tpl = {wid for wid in set(wids) if np.clip(z[wid] * a.scale, 0, 1).sum() >= a.min_mass}
    for i, (wid, t, lid) in enumerate(zip(wids, ts, lids)):
        T, links = meta[wid]
        if wid in keep_tpl:
            new[i] = old_vals[i]
            continue
        if wid not in cache:
            pr = np.clip(z[wid] * a.scale, 0, 1)
            if a.min_mass > 0 and pr.sum() < a.min_mass:  # an onset window guarantees a queue: lift low-mass vectors
                pr = np.clip(pr * (a.min_mass / max(pr.sum(), 1e-9)), 0, 1)
            queue_train.RNG = np.random.default_rng(zlib.crc32(wid.encode()))  # per-window stream: a window's decode does not
            cache[wid] = decode(pr, a.mode, thr=a.thr)  # depend on which other windows are present
        new[i] = int(t - T == 6 and cache[wid][links[lid]])
    old = pred.loc[sel, "queue_pred"].to_numpy()
    print(
        f"onset {a.mode} {a.thr if a.mode == 'thr' else ''}: cells changed {int((old != new).sum())}, positives {int(old.sum())} -> {int(new.sum())}"
    )
    pred.loc[sel, "queue_pred"] = new
    pred.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
