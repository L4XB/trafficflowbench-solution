"""Materialise the selector-matched training windows and the queue truth for every panel.

Outputs: cache_top3/<panel>/train_queued.npy (bool, raw filled truth), train_speed_filled.npy,
         results_top3/queue/windows_train.csv (panel, d, T, condition, piou, n30, hor_n, date, month)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from tfb.common import CACHE, PROJECT, QUEUE_PANELS, load_panel
from tfb.labels.selector import merged_greedy, validity
from tfb.queue.data import filled_speed, queued_observed

TRUTH = "raw"


def one(panel: str, TRUTH: str = "raw") -> pd.DataFrame:
    """Train windows of one panel: the merged and the per-condition greedy selections, deduplicated per (d, T, condition).

    Saves the panel's queue truth and validity table (and, for the raw truth, the filled speed) on the way.
    """
    t0 = time.time()
    p = load_panel(panel, "train")
    suf = "" if TRUTH == "raw" else f"_{TRUTH}"
    if TRUTH == "raw":
        sf = filled_speed(p)
        np.save(CACHE / panel / "train_speed_filled.npy", sf.astype(np.float32))
    else:
        sf = np.load(CACHE / panel / f"train_speed_{TRUTH}.npy")
    truth = sf <= p.param("v_cut")[None, None, :]
    np.save(CACHE / panel / f"train_queued{suf}.npy", truth)
    v = validity(truth, queued_observed(p), p["elig"])
    v.to_parquet(PROJECT / "results_top3" / "queue" / f"validity_{panel}{suf}.parquet")
    parts = [merged_greedy(v, quota=None).assign(pass_="merged")]
    for cond in ("onset", "ongoing"):
        vv = v.copy()
        vv["onset_ok"] = vv.onset_ok & (cond == "onset")
        vv["ongoing_ok"] = vv.ongoing_ok & (cond == "ongoing")
        parts.append(merged_greedy(vv, quota=None).assign(pass_=cond))
    g = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(["d", "T", "condition"], keep="first")
        .sort_values(["d", "T"])
        .reset_index(drop=True)
    )
    g["panel"] = panel
    g["date"] = [p.dates[d] for d in g.d]
    g["month"] = g.date.str.slice(0, 7)
    print(
        f"{panel}: {len(g)} windows ({(g.condition == 'onset').sum()} onset, {(g.condition == 'ongoing').sum()} ongoing) in {time.time() - t0:.0f}s",
        flush=True,
    )
    return g


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default="raw")
    a = ap.parse_args()
    TRUTH = a.truth
    from functools import partial

    with ProcessPoolExecutor(4) as ex:
        parts = list(ex.map(partial(one, TRUTH=TRUTH), QUEUE_PANELS))
    w = pd.concat(parts, ignore_index=True)
    w.to_csv(PROJECT / "results_top3" / "queue" / f"windows_train{'' if TRUTH == 'raw' else '_' + TRUTH}.csv", index=False)
    print(w.groupby(["panel", "condition"]).size().unstack())
    print(w.groupby(["month", "condition"]).size().unstack())
