"""Build the array cache: one process per panel, all three splits.

Run: python -m tfb.cache.build [--panels P ...]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from tfb.common import CACHE, PANELS, RELEASE, SPLITS, STEPS, t_index

MAIN_COLS = ["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]
RAMP_COLS = ["date", "timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"]


def read_days(files: list[Path], cols: list[str]) -> pd.DataFrame:
    """The columns `cols` of the parquet day files, concatenated into one DataFrame."""
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)


def build_panel(panel: str) -> str:
    """Cache all splits of one panel: (days, 288, links) masked layer, pct, eligibility and Task-1 targets, ramp arrays, meta.json.

    The train split also gets the unmasked truth layer; returns a one-line summary.
    """
    t0 = time.time()
    pdir = RELEASE / "corridors" / panel
    out = CACHE / panel
    out.mkdir(parents=True, exist_ok=True)
    topo = pd.read_csv(pdir / "network" / "lwr_mainline_topology.csv", dtype=str)
    topo["order_index"] = topo.order_index.astype(int)
    topo = topo.sort_values("order_index")
    links = topo.link_id.tolist()
    lidx = {l: i for i, l in enumerate(links)}
    fd = pd.read_csv(pdir / "network" / "fd_parameters.csv").set_index("link_id").loc[links]
    lk = pd.read_csv(pdir / "network" / "links.csv").set_index("link_id").loc[links]
    params = {
        "length_km": fd.length_km.tolist(),
        "lanes": fd.lanes.tolist(),
        "free_speed": fd.free_speed_kmh.tolist(),
        "capacity": fd.capacity_vph.tolist(),
        "k_crit": fd.critical_density.tolist(),
        "k_jam": fd.k_jam.tolist(),
        "free_speed_links": lk.free_speed_kmh.tolist(),
        "v_cut": (0.6 * lk.free_speed_kmh).tolist(),
        "topo_free_speed": topo.free_speed_kmh.astype(float).tolist(),
        "topo_capacity": topo.capacity_vph.astype(float).tolist(),
    }
    # ramps
    amap = pd.read_csv(pdir / "network" / "ramp_attachment_map.csv", dtype=str)
    ramps = amap.ramp_link_id.tolist()
    ridx = {r: i for i, r in enumerate(ramps)}
    ramp_meta = {
        "ramp_ids": ramps,
        "ramp_type": amap.ramp_type.tolist(),
        "ramp_link": [lidx.get(l, -1) for l in amap.nearest_mainline_link_id],
    }
    meta = {"panel": panel, "links": links, "stations": {}, "params": params, "ramps": ramp_meta, "dates": {}, "regimes": {}}
    for split in SPLITS:
        masked_files = sorted((pdir / split / "mainline_states_masked").glob("**/*.parquet"))
        tpl = pacsv.read_csv(
            RELEASE / "task1" / panel / split / "sample_submission_state.csv",
            convert_options=pacsv.ConvertOptions(
                include_columns=["timestamp", "link_id", "mask_regime"],
                column_types={"timestamp": pa.string(), "link_id": pa.string(), "mask_regime": pa.string()},
            ),
        ).to_pandas()
        tdate = tpl.timestamp.str.slice(0, 10)
        regime = {}
        for f in masked_files:
            d = f.stem.replace("synthetic_mainline_", "").replace("_", "-")
            regime[d] = f.parent.name.split("=")[-1]
        for d, rg in zip(tdate, tpl.mask_regime):  # a release cut at a Task-2 origin may lack later day files entirely
            regime.setdefault(d, rg)
        dates = sorted(regime)
        didx = {d: i for i, d in enumerate(dates)}
        meta["dates"][split] = dates
        meta["regimes"][split] = [regime[d] for d in dates]
        D, L = len(dates), len(links)
        m = read_days(masked_files, MAIN_COLS)
        di = m.date.map(didx).to_numpy()
        ti = t_index(m.timestamp.to_numpy())
        li = m.link_id.map(lidx).to_numpy()
        assert not np.isnan(li.astype(float)).any(), "unknown link in masked layer"
        if split == "train":
            st = m.drop_duplicates("link_id").set_index("link_id").station_id
            meta["stations"] = {l: st[l] for l in links}
        shape = (D, STEPS, L)
        for col, key in (("speed_kmh", "speed"), ("flow_vph", "flow"), ("occupancy", "occ")):
            a = np.full(shape, np.nan, dtype=np.float32)
            a[di, ti, li] = m[col].to_numpy(dtype=np.float32)
            np.save(out / f"{split}_{key}_m.npy", a)
        pct = np.zeros(shape, dtype=np.int8)
        pct[di, ti, li] = m.pct_observed.to_numpy(dtype=np.int8)
        np.save(out / f"{split}_pct.npy", pct)
        elig = np.zeros(shape, dtype=bool)
        elig[di, ti, li] = m.is_score_eligible.to_numpy(dtype=bool)
        np.save(out / f"{split}_elig.npy", elig)
        del m
        if split == "train":
            truth_files = sorted((pdir / split / "mainline_states").glob("**/*.parquet"))
            u = read_days(truth_files, MAIN_COLS)
            di = u.date.map(didx).to_numpy()
            ti = t_index(u.timestamp.to_numpy())
            li = u.link_id.map(lidx).to_numpy()
            for col, key in (("speed_kmh", "speed"), ("flow_vph", "flow"), ("occupancy", "occ")):
                a = np.full(shape, np.nan, dtype=np.float32)
                a[di, ti, li] = u[col].to_numpy(dtype=np.float32)
                np.save(out / f"{split}_{key}.npy", a)
            del u
        # Task 1 template targets
        tdi = tdate.map(didx).to_numpy()
        tti = t_index(tpl.timestamp.to_numpy())
        tli = tpl.link_id.map(lidx).to_numpy()
        target = np.zeros(shape, dtype=bool)
        target[tdi, tti, tli] = True
        assert target.sum() == len(tpl), (target.sum(), len(tpl))
        # regime consistency: template regime == day regime
        reg_day = np.asarray(meta["regimes"][split])
        assert (tpl.mask_regime.to_numpy() == reg_day[tdi]).all(), "template regime != day regime"
        np.save(out / f"{split}_target.npy", target)
        # ramps
        rfiles = sorted((pdir / split / "ramp_states").glob("**/*.parquet"))
        r = read_days(rfiles, RAMP_COLS)
        rdi = r.date.map(didx).to_numpy()
        rti = t_index(r.timestamp.to_numpy())
        rri = r.ramp_link_id.map(ridx).to_numpy()
        rshape = (D, STEPS, len(ramps))
        rf = np.full(rshape, np.nan, dtype=np.float32)
        rf[rdi, rti, rri] = r.flow_vph.to_numpy(dtype=np.float32)
        np.save(out / f"{split}_ramp_flow.npy", rf)
        re = np.zeros(rshape, dtype=bool)
        re[rdi, rti, rri] = r.is_score_eligible.to_numpy(dtype=bool) & (r.pct_observed.to_numpy() >= 75)
        np.save(out / f"{split}_ramp_elig.npy", re)
        del r
    (out / "meta.json").write_text(json.dumps(meta))
    return f"{panel}: {len(links)} links, {len(ramps)} ramps, {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(build_panel, a.panels):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
