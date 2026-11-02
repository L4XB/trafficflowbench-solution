"""Compare the queue predictions of a truncated-release run with the full run for every window whose forecast origin is at or
before the cut (the windows the organizers' check covers; later windows keep their full history in the check)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--trunc", required=True)
    ap.add_argument("--cut", required=True)
    ap.add_argument("--data", default="data/kaggle_public", help="full release; its window_index.csv files give the forecast origins")
    a = ap.parse_args()
    cut = a.cut.split()[0]
    idx = pd.concat([pd.read_csv(p, usecols=["window_id", "forecast_origin"]) for p in Path(a.data).glob("task2/*/*/window_index.csv")])
    origin = idx.set_index("window_id").forecast_origin
    f = pd.read_csv(a.full)
    t = pd.read_csv(a.trunc)
    keys = ["window_id", "timestamp", "link_id"]
    m = t.merge(f, on=keys, how="left", suffixes=("_t", "_f"), validate="one_to_one")
    assert m.queue_pred_f.notna().all(), "truncated run has cells the full run does not"
    assert m.window_id.isin(origin.index).all(), "window without a forecast origin in the release"
    later = m.window_id.map(origin) > cut
    n_later = m.loc[later, "window_id"].nunique()
    m = m[~later]
    m["diff"] = m.queue_pred_t != m.queue_pred_f
    per = m.groupby("window_id")["diff"].sum()
    split = m.window_id.str.contains("_private_").map({True: "private", False: "validation"})
    print(
        f"cut {a.cut}: windows compared {per.size} (private {m[split == 'private'].window_id.nunique()}; {n_later} later windows skipped), "
        f"cells {len(m)}, differing cells {int(m['diff'].sum())}, windows with differences {int((per > 0).sum())}"
    )
    if (per > 0).any():
        print(per[per > 0].to_string())


if __name__ == "__main__":
    main()
