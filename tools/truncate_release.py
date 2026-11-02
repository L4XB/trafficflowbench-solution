"""Copy of the release with every row of one split stamped after a cut time removed (the organizers' Task-2 check).

Rewritten: <split>/mainline_states_masked and <split>/ramp_states day files (rows with timestamp > cut dropped; later days become
empty files with the same schema), task2/<panel>/<split>/window_history.parquet (rows after the cut) and window_index.csv (windows
whose forecast origin is after the cut). Everything else -- the other splits, networks, task-4 files and the submission templates --
is symlinked unchanged.
Run: python tools/truncate_release.py --src data/kaggle_public --dst /tmp/trunc/data/kaggle_public --split private --cut 2031-04-05T08:00:00Z
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def cut_table(t: pa.Table, cut: str) -> pa.Table:
    """Rows of `t` with timestamp <= cut (the cut cast to the column's type)."""
    return t.filter(pc.less_equal(t.column("timestamp"), pa.scalar(cut, t.schema.field("timestamp").type)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--split", default="private")
    ap.add_argument("--cut", required=True, help="ISO time, e.g. 2031-04-05T08:00:00Z")
    ap.add_argument(
        "--mode", default="empty", choices=["empty", "delete"], help="days after the cut: empty files (same schema) or no file at all"
    )
    a = ap.parse_args()
    src, dst = Path(a.src).resolve(), Path(a.dst)
    n_rows = {"kept": 0, "dropped": 0}
    for root, dirs, files in os.walk(src):
        r = Path(root)
        rel = r.relative_to(src)
        out = dst / rel
        out.mkdir(parents=True, exist_ok=True)
        in_split = len(rel.parts) >= 3 and rel.parts[0] == "corridors" and rel.parts[2] == a.split
        in_task2 = len(rel.parts) == 3 and rel.parts[0] == "task2" and rel.parts[2] == a.split
        for f in files:
            s, o = r / f, out / f
            if o.exists() or o.is_symlink():
                o.unlink()
            if in_split and f.endswith(".parquet"):
                t = pq.read_table(s)
                k = cut_table(t, a.cut)
                n_rows["kept"] += k.num_rows
                n_rows["dropped"] += t.num_rows - k.num_rows
                if k.num_rows == 0 and a.mode == "delete":
                    continue
                pq.write_table(k, o)
            elif in_task2 and f == "window_history.parquet":
                t = pq.read_table(s)
                pq.write_table(cut_table(t, a.cut), o)
            elif in_task2 and f == "window_index.csv":
                w = pd.read_csv(s)
                w[w.forecast_origin <= a.cut].to_csv(o, index=False)
            else:
                o.symlink_to(s)
    print(f"cut {a.cut} ({a.split}): mainline/ramp rows kept {n_rows['kept']}, dropped {n_rows['dropped']} -> {dst}")


if __name__ == "__main__":
    main()
