"""Assemble a full submission from block sources on top of the anchor file.

    python -m tfb.assemble --out results_top3/subs/NAME.csv \
        [--state-dir results_top3/state/MODEL] [--queue results_top3/queue/pred_MODEL.csv] [--odme path.csv]

Rows not covered by a block source keep the anchor's values. Every block is validated.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from tfb.common import PANELS, RELEASE, load_panel, t_index

ANCHOR = RELEASE / "sample_submission.csv"
KEY = RELEASE / "submission_key.csv"
SPEED_CAP = 120.75


def load_key() -> pd.DataFrame:
    """The release's submission_key.csv: submission_id and the ten row-key columns (the latter read as strings)."""
    cols = [
        "submission_id",
        "task",
        "split",
        "panel",
        "timestamp",
        "station_id",
        "link_id",
        "mask_regime",
        "window_id",
        "departure_time",
        "path_id",
    ]
    t = pacsv.read_csv(
        KEY, convert_options=pacsv.ConvertOptions(include_columns=cols, column_types={c: pa.string() for c in cols if c != "submission_id"})
    )
    return t.to_pandas()


def load_anchor(path: Path = ANCHOR) -> pd.DataFrame:
    """Anchor submission (default: the release's sample_submission.csv) with float prediction columns and an integer queue_pred."""
    sub = pacsv.read_csv(path).to_pandas()
    for c in ("speed_kmh", "flow_vph", "path_flow"):  # sample_submission.csv ships integer zeros
        sub[c] = sub[c].astype(np.float64)
    sub["queue_pred"] = sub["queue_pred"].astype(np.int64)
    return sub


def apply_state(sub: pd.DataFrame, key: pd.DataFrame, state_dir: Path) -> int:
    """state_dir: <panel>_<split>.npz with speed and flow (days, 288, links); replaces the matching state rows, returns their count.

    Speeds are clipped to [0, SPEED_CAP], negative flows set to 0; panel/split pairs without a file keep the anchor's rows.
    """
    n = 0
    is_state = (key.task == "state").to_numpy()
    for split in ("validation", "private"):
        for panel in PANELS:
            f = state_dir / f"{panel}_{split}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            sp, fl = z["speed"], z["flow"]
            p = load_panel(panel, split)
            m = is_state & (key.split.to_numpy() == split) & (key.panel.to_numpy() == panel)
            idx = np.where(m)[0]
            ts = key.timestamp.to_numpy()[idx]
            di = pd.Series([s[:10] for s in ts]).map({d: i for i, d in enumerate(p.dates)}).to_numpy()
            ti = t_index(ts)
            li = pd.Series(key.link_id.to_numpy()[idx]).map({l: i for i, l in enumerate(p.links)}).to_numpy()
            assert not np.isnan(di.astype(float)).any() and not np.isnan(li.astype(float)).any(), (panel, split)
            v = sp[di, ti, li]
            q = fl[di, ti, li]
            assert np.isfinite(v).all() and np.isfinite(q).all(), (panel, split, "non-finite state prediction")
            sub.loc[idx, "speed_kmh"] = np.clip(v, 0.0, SPEED_CAP)
            sub.loc[idx, "flow_vph"] = np.maximum(q, 0.0)
            n += len(idx)
    return n


def apply_queue(sub: pd.DataFrame, key: pd.DataFrame, pred: pd.DataFrame) -> int:
    """pred: window_id, timestamp, link_id, queue_pred (0/1) for any subset of windows."""
    is_q = key.task == "queue"
    kq = key.loc[is_q, ["window_id", "timestamp", "link_id"]].copy()
    kq["row"] = kq.index
    m = kq.merge(pred, on=["window_id", "timestamp", "link_id"], how="inner", validate="one_to_one")
    assert m.queue_pred.isin([0, 1]).all(), "queue_pred must be binary"
    sub.loc[m.row.to_numpy(), "queue_pred"] = m.queue_pred.to_numpy().astype(int)
    covered = m.groupby("window_id").size()
    full = kq.groupby("window_id").size().loc[covered.index]
    assert (covered == full).all(), "a window was only partially covered"
    return len(m)


def validate(sub: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Assert the row count, the ids, no NaN and the queue / odme ranges (warn on the state range); return summary statistics."""
    assert len(sub) == 6980503 and (sub.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    assert sub[["speed_kmh", "flow_vph", "queue_pred", "path_flow"]].notna().all().all(), "NaN in submission"
    st = sub[key.task == "state"]
    qu = sub[key.task == "queue"]
    od = sub[key.task == "odme"]
    if not ((st.speed_kmh >= 0).all() and (st.speed_kmh <= SPEED_CAP + 1e-6).all() and (st.flow_vph >= 0).all()):
        print(
            f"WARNING state range: speed [{st.speed_kmh.min():.2f}, {st.speed_kmh.max():.2f}] flow [{st.flow_vph.min():.2f}, {st.flow_vph.max():.2f}]"
        )
    assert qu.queue_pred.isin([0, 1]).all() and (od.path_flow >= 0).all()
    return {
        "rows": len(sub),
        "state_mean_speed": float(st.speed_kmh.mean()),
        "state_mean_flow": float(st.flow_vph.mean()),
        "queue_positives": int(qu.queue_pred.sum()),
        "odme_sum": float(od.path_flow.sum()),
    }


def write(sub: pd.DataFrame, out: Path) -> str:
    """Write the submission as .csv (suffix forced) and a .zip holding it; returns the CSV's sha256 prefix (16 hex digits)."""
    out = out if out.suffix == ".csv" else out.with_suffix(".csv")  # Kaggle errors on an archive entry without .csv
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, float_format="%.6f")
    with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out, arcname=out.name)
    return hashlib.sha256(out.read_bytes()).hexdigest()[:16]


def apply_odme(sub: pd.DataFrame, key: pd.DataFrame, frames: list[pd.DataFrame]) -> int:
    """frames: path-flow tables keyed by (panel, departure_time, path_id); replaces the matching odme rows."""
    od = pd.concat(frames, ignore_index=True)
    assert not od.duplicated(["panel", "departure_time", "path_id"]).any(), "duplicate odme keys"
    is_od = (key.task == "odme").to_numpy()
    idx = np.where(is_od)[0]
    k = pd.DataFrame(
        {"panel": key.panel.to_numpy()[idx], "departure_time": key.departure_time.to_numpy()[idx], "path_id": key.path_id.to_numpy()[idx]}
    )
    m = k.merge(od[["panel", "departure_time", "path_id", "path_flow"]], on=["panel", "departure_time", "path_id"], how="left")
    hit = m.path_flow.notna().to_numpy()
    assert (m.path_flow[hit] >= 0).all(), "negative path flow"
    sub.loc[idx[hit], "path_flow"] = m.path_flow.to_numpy()[hit]
    return int(hit.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-dir")
    ap.add_argument("--queue")
    ap.add_argument("--anchor", default=str(ANCHOR))
    ap.add_argument(
        "--odme",
        nargs="*",
        help="path-flow tables (panel, departure_time, path_id, path_flow); together they must cover every odme row they touch",
    )
    a = ap.parse_args()
    t0 = time.time()
    key = load_key()
    sub = load_anchor(Path(a.anchor))
    assert (sub.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    print(f"loaded key+anchor ({time.time() - t0:.0f}s)")
    if a.state_dir:
        print("state rows replaced:", apply_state(sub, key, Path(a.state_dir)), f"({time.time() - t0:.0f}s)")
    if a.queue:
        print("queue rows replaced:", apply_queue(sub, key, pd.read_csv(a.queue)), f"({time.time() - t0:.0f}s)")
    if a.odme:
        print("odme rows replaced:", apply_odme(sub, key, [pd.read_csv(f) for f in a.odme]), f"({time.time() - t0:.0f}s)")
    print(validate(sub, key))
    print("sha256[:16]", write(sub, Path(a.out)), f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
