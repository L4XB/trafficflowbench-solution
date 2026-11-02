"""Compare a submission built by run_all.sh with a reference file (for example the file submitted on Kaggle), block by block.

State rows: speed and flow differences; queue rows: differing cells per month; ODME rows: path-flow differences.
    python tools/compare_submission.py --ours submission/submission_A.csv --ref final_A_cand_lFbo2bs_qfswap_kcp11.zip
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

COLS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def read(path: Path) -> dict[str, np.ndarray]:
    """A submission CSV, or a zip archive that holds one, as numpy columns."""
    opts = pacsv.ConvertOptions(include_columns=COLS, column_types={"task": pa.string()})
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            (name,) = [n for n in z.namelist() if n.endswith(".csv")]
            t = pacsv.read_csv(io.BytesIO(z.read(name)), convert_options=opts)
    else:
        t = pacsv.read_csv(path, convert_options=opts)
    return {c: t.column(c).to_numpy() for c in COLS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--key", default="data/kaggle_public/submission_key.csv", type=Path)
    a = ap.parse_args()
    o, r = read(a.ours), read(a.ref)
    assert (o["submission_id"] == r["submission_id"]).all(), "row order differs"
    split = pacsv.read_csv(a.key, convert_options=pacsv.ConvertOptions(include_columns=["split"])).column("split").to_numpy(zero_copy_only=False)
    task = o["task"].astype(str)
    st, qu, od = task == "state", task == "queue", task == "odme"
    for c in ("speed_kmh", "flow_vph"):
        d = np.abs(o[c][st] - r[c][st])
        print(f"state {c:9s}: rows {st.sum():,}, differing by > 1e-3 {int((d > 1e-3).sum()):,}, max {d.max():.4f}, rms {np.sqrt((d**2).mean()):.2e}")
    for sp, month in (("validation", "March"), ("private", "April")):
        m = qu & (split == sp)
        print(f"queue {month:5s}      : cells {m.sum():,}, differing {int((o['queue_pred'][m] != r['queue_pred'][m]).sum())}")
    d = np.abs(o["path_flow"][od] - r["path_flow"][od])
    print(f"odme path_flow   : rows {od.sum():,}, differing by > 1e-6 {int((d > 1e-6).sum()):,}, max {d.max():.2e}")


if __name__ == "__main__":
    main()
