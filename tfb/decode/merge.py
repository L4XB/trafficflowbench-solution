"""Merge queue prediction tables per condition: --onset FILE takes the onset windows, --ongoing FILE the ongoing ones.

Run: python -m tfb.decode.merge --onset onset_pred.csv --ongoing ongoing_pred.csv --out pred.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from tfb.common import QUEUE_PANELS
from tfb.queue.data import released_windows


def window_conditions() -> pd.Series:
    """window_id -> condition for every released evaluation window."""
    parts = []
    for pn in QUEUE_PANELS:
        for split in ("validation", "private"):
            w = released_windows(pn, split)
            parts.append(w[["window_id", "condition"]])
    w = pd.concat(parts, ignore_index=True)
    return w.set_index("window_id").condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onset", required=True)
    ap.add_argument("--ongoing", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cond = window_conditions()
    on = pd.read_csv(a.onset)
    og = pd.read_csv(a.ongoing)
    on = on[on.window_id.map(cond) == "queue_onset"]
    og = og[og.window_id.map(cond) == "queue_ongoing"]
    out = pd.concat([on, og], ignore_index=True)
    assert out.window_id.nunique() == cond.size, (out.window_id.nunique(), cond.size)
    assert not out.duplicated(["window_id", "timestamp", "link_id"]).any()
    out.to_csv(a.out, index=False)
    print(
        f"onset rows {len(on)} (positives {int(on.queue_pred.sum())}), ongoing rows {len(og)} (positives {int(og.queue_pred.sum())}) -> {a.out}"
    )


if __name__ == "__main__":
    main()
