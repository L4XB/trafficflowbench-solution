"""Free-flow level correction of the Task-1 speeds: the last state stage, after the U-Net ensemble and the blackout corrector.

In free flow the generator draws a link's speed as a per-link constant plus white noise that is partly common to the whole panel at
one step. The constant is the same in every train month but differs in March and April, and a U-Net that sees one panel-day has to
re-estimate it from that day. The masked layer of the whole month pins it down. For every hidden cell that the base state places in
free flow (at most 3 km/h below the link's month level) and whose step has at least ten visible free-flow cells in the panel:

    v = w * v_base + (1 - w) * min(level_l + c_t, 120.7),   w = 0.15 (0.35 when v_base is 2-3 km/h below the level),

level_l = median of the link's observed speeds within 6 km/h of its 75th percentile over the split (one month), c_t = mean deviation
of the panel's visible free-flow cells from their levels at step t. Flow is unchanged, and the blanked spans after the Task-2 windows
(no visible cell at those steps) keep the blackout corrector's values. Task 1 is offline, so the whole month may be used; no Task-2
input reads these values.

Run: python -m tfb.state.level_apply --base enslongF_bo2bs --out enslongF_bo2bs_lvl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tfb.common import PANELS, PROJECT, load_panel

ST = PROJECT / "results_top3" / "state"
W_FREE, W_EDGE, BAND, EDGE, CAP, MIN_VISIBLE = 0.15, 0.35, 3.0, 2.0, 120.7, 10


def month_level(vm: np.ndarray) -> np.ndarray:
    """Per-link free-flow level from a masked month (D, T, L): median of the observed speeds near the link's upper range."""
    lev = np.full(vm.shape[2], np.nan)
    for l in range(vm.shape[2]):
        x = vm[:, :, l][np.isfinite(vm[:, :, l])]
        if x.size < 200:
            continue
        lev[l] = np.median(x[x > np.percentile(x, 75) - 6.0])
    return lev


def correct_day(v: np.ndarray, vm: np.ndarray, lev: np.ndarray, target: np.ndarray) -> np.ndarray:
    """One day: v (T, L) base speeds, vm (T, L) masked layer, lev (L,) month level, target (T, L) hidden cells to fill."""
    dev = vm - lev[None, :]
    free = np.isfinite(dev) & (np.abs(dev) < 5.0)
    cnt = free.sum(1)
    c = np.where(cnt >= MIN_VISIBLE, np.where(free, dev, 0.0).sum(1) / np.maximum(cnt, 1), 0.0)
    est = np.minimum(lev[None, :] + c[:, None], CAP)
    d = v - lev[None, :]
    sel = target & np.isfinite(lev)[None, :] & (d >= -BAND) & (cnt >= MIN_VISIBLE)[:, None]
    w = np.where(d >= -EDGE, W_FREE, W_EDGE)
    out = v.copy()
    out[sel] = w[sel] * v[sel] + (1 - w[sel]) * est[sel]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    a_ = ap.parse_args()
    out_dir = ST / a_.out
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in a_.splits:
        for panel in PANELS:
            p = load_panel(panel, split)
            z = np.load(ST / a_.base / f"{panel}_{split}.npz")
            v = z["speed"].astype(np.float64)
            vm = np.asarray(p["speed_m"], dtype=np.float64)                 # (D, T, L) released masked layer
            target = np.asarray(p["target"]) & np.asarray(p["elig"])
            lev = month_level(vm)
            v2 = np.stack([correct_day(v[k], vm[k], lev, target[k]) for k in range(v.shape[0])])
            np.savez(out_dir / f"{panel}_{split}.npz", speed=v2.astype(z["speed"].dtype), flow=z["flow"])
            print(f"{split} {panel}: {int((np.abs(v2 - v) > 1e-9).sum())} of {int(target.sum())} hidden cells moved", flush=True)
    (out_dir / "predict_info.json").write_text(json.dumps({"base": a_.base, "out": a_.out, "rule": "free-flow level", "w_free": W_FREE,
        "w_edge": W_EDGE, "band": BAND, "edge": EDGE, "cap": CAP, "min_visible": MIN_VISIBLE}, indent=1))


if __name__ == "__main__":
    main()
