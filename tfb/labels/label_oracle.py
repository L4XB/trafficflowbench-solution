"""Which label version is closest to the organizer's noise-free truth? Oracle = the 80 released train windows.

For each candidate speed field, run the selector replica (validity + merged greedy, quota 5) and count exact reproductions
of the released windows, and released windows that are invalid under that field ("violations").
Run: python -m tfb.labels.label_oracle
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from tfb.common import CACHE, QUEUE_PANELS, STEPS, load_panel
from tfb.labels.selector import merged_greedy, validity
from tfb.queue.data import filled_speed, queued_observed, released_windows


def fields(pn: str) -> dict[str, np.ndarray]:
    """Candidate train speed fields (days, 288, links) per label version in ORACLE_VERSIONS (default raw, dn, post, mix0.3, mix0.6)."""
    p = load_panel(pn, "train")
    obs = np.asarray(p["speed"], np.float64)
    dn = np.asarray(np.load(CACHE / pn / "train_speed_dn.npy"), np.float64)
    post = np.asarray(np.load(CACHE / pn / "train_speed_post.npy"), np.float64)
    fin = np.isfinite(obs)
    import os

    versions = os.environ.get("ORACLE_VERSIONS", "raw,dn,post,mix0.3,mix0.6").split(",")
    out = {}
    night = obs[:, :60]
    m = np.nanmean(night, axis=1, keepdims=True)
    s = float(np.nanmedian(np.sqrt(np.nanmean((night - m) ** 2, axis=1))))
    for name in versions:
        if name == "raw":
            out[name] = filled_speed(p).astype(np.float64)
        elif name == "dn":
            out[name] = dn
        elif name == "post":
            out[name] = post
        elif name == "obsdn2":  # observation where present, long-member U-Net field in the gaps
            dn2 = np.asarray(np.load(CACHE / pn / "train_speed_dn2.npy"), np.float64)
            out[name] = np.where(fin, obs, dn2)
        elif name.startswith("mixdn2_"):  # w * obs + (1 - w) * dn2
            w = float(name[7:])
            dn2 = np.asarray(np.load(CACHE / pn / "train_speed_dn2.npy"), np.float64)
            out[name] = np.where(fin, w * np.nan_to_num(obs) + (1 - w) * dn2, dn2)
        elif name.startswith("mix"):
            w = float(name[3:])
            out[name] = np.where(fin, w * np.nan_to_num(obs) + (1 - w) * dn, dn)
        elif name.startswith("rob"):  # robust mixture posterior: rob<e1>_<e2>_<pi2>
            e1, e2, pi2 = (float(x) for x in name[3:].split("_"))
            r = np.nan_to_num(obs - dn)
            sd1 = np.sqrt(s**2 + e1**2)
            sd2 = np.sqrt(s**2 + e2**2)
            n1 = (1 - pi2) / sd1 * np.exp(-0.5 * (r / sd1) ** 2)
            n2 = pi2 / sd2 * np.exp(-0.5 * (r / sd2) ** 2)
            resp2 = n2 / (n1 + n2)
            x1 = (np.nan_to_num(obs) / s**2 + dn / e1**2) / (1 / s**2 + 1 / e1**2)
            x2 = (np.nan_to_num(obs) / s**2 + dn / e2**2) / (1 / s**2 + 1 / e2**2)
            out[name] = np.where(fin, (1 - resp2) * x1 + resp2 * x2, dn)
    return out


def one(pn: str) -> list[dict]:
    """One row per label version of the panel: exact and within-2-step reproductions of the released train windows, and violations."""
    t0 = time.time()
    p = load_panel(pn, "train")
    vc = p.param("v_cut")[None, None, :]
    obs_flag = queued_observed(p)
    elig = p["elig"]
    w = released_windows(pn, "train")
    rel = sorted([(p.dates.index(r.date), int(r.T), r.condition[6:]) for r in w.itertuples()], key=lambda x: x[0] * STEPS + x[1])
    rows = []
    for name, sp in fields(pn).items():
        truth = sp <= vc
        v = validity(truth, obs_flag, elig)
        g = merged_greedy(v, quota=5)
        got = [(r.d, r.T, r.condition) for r in g.itertuples()]
        exact = sum(1 for a, b in zip(got, rel) if a == b)
        near = sum(1 for a, b in zip(got, rel) if a[2] == b[2] and abs(a[0] * STEPS + a[1] - b[0] * STEPS - b[1]) <= 2)
        vi = v.set_index(["d", "T"])
        viol = 0
        for d, T, cond in rel:
            ok = vi.loc[(d, T), "onset_ok" if cond == "onset" else "ongoing_ok"] if (d, T) in vi.index else False
            viol += int(not bool(ok))
        rows.append({"panel": pn, "label": name, "exact": exact, "within2": near, "violations": viol})
    print(f"{pn} done ({time.time() - t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex:
        parts = list(ex.map(one, QUEUE_PANELS))
    df = pd.DataFrame([r for part in parts for r in part])
    print(df.pivot(index="panel", columns="label", values="exact").to_string())
    print("\nTOTAL over 80 released windows:")
    print(df.groupby("label")[["exact", "within2", "violations"]].sum().to_string())
