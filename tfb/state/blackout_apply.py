"""Apply a blackout corrector to the evaluation splits and write a new state directory for the assembler.

For every released Task-2 window (d, T) of a queue panel the masked layer is blank on all links for T+1..T+18. Inside that span
the base state (e.g. enslongF) is replaced by feed + corrector, where feed is the U-Net prediction the corrector was trained on
(same members, made with the real masked layer, i.e. with the blackouts) -- or blended: w * corrected + (1 - w) * base.
Every other cell, and the non-queue panels, keep the base state.

Run: python -m tfb.state.blackout_apply --base enslongF --feed pred_long46 --corr bo_stack_long46_b --out enslongF_bo46b
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from tfb.common import PANELS, PROJECT, QUEUE_PANELS, load_panel
from tfb.queue.data import released_windows
from tfb.state.blackout_stack import BLACK, FEATS, ramp_link_flows, smooth_time, window_features

ST = PROJECT / "results_top3" / "state"


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--feed", required=True)
    ap.add_argument("--corr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--splits", nargs="*", default=["validation", "private"])
    ap.add_argument("--smooth", type=int, default=0, help="passes of [1/4,1/2,1/4] smoothing of the corrections along time")
    a_ = ap.parse_args()
    cdir = ST / a_.corr
    use = json.loads((cdir / "meta.json").read_text())["feats"]
    cols = [FEATS.index(f) for f in use]
    boosters = [lgb.Booster(model_file=str(cdir / "speed.txt")), lgb.Booster(model_file=str(cdir / "flow.txt"))]
    out = ST / a_.out
    out.mkdir(parents=True, exist_ok=True)
    for sp in a_.splits:
        for pn in PANELS:
            src = ST / a_.base / f"{pn}_{sp}.npz"
            if pn not in QUEUE_PANELS:
                shutil.copyfile(src, out / f"{pn}_{sp}.npz")
                continue
            base = np.load(src)
            bv, bq = base["speed"].copy(), base["flow"].copy()
            feed = np.load(ST / a_.feed / f"{pn}_{sp}.npz")
            fv, fq = feed["speed"].astype(np.float32), feed["flow"].astype(np.float32)
            p = load_panel(pn, sp)
            vm = np.asarray(p["speed_m"], dtype=np.float32)
            qm = np.asarray(p["flow_m"], dtype=np.float32)
            lanes = p.param("lanes").astype(np.float32)
            reg = np.asarray(p.regimes)
            r = released_windows(pn, sp)
            ron, roff, ron_ok, roff_ok = ramp_link_flows(p)
            n_cells = 0
            dv = []
            for dt, T, cond in zip(r.date, r["T"].astype(int), r.condition):
                d = p.dates.index(dt)
                sl = slice(T + 1, T + 1 + BLACK)
                assert np.isnan(vm[d, sl]).all(), (pn, sp, dt, T)  # the blackout is where we expect it
                obs = np.isfinite(vm[d])
                F = window_features(
                    p,
                    pn,
                    vm[d],
                    qm[d] / lanes,
                    obs,
                    fv[d],
                    fq[d] / lanes[None, :],
                    int(T),
                    pd.Timestamp(dt).weekday(),
                    str(reg[d]),
                    (ron[d], roff[d], ron_ok[d], roff_ok[d]),
                    float(cond == "queue_onset"),
                )
                X = F.reshape(-1, F.shape[-1])[:, cols]
                cv = boosters[0].predict(X, num_threads=6).reshape(BLACK, -1)
                cq = boosters[1].predict(X, num_threads=6).reshape(BLACK, -1)
                cv, cq = smooth_time(cv, a_.smooth), smooth_time(cq, a_.smooth)
                cvv = np.clip(fv[d, sl] + cv, 0, 130)
                cqq = np.clip(fq[d, sl] + cq * lanes[None, :], 0, None)
                nv = a_.w * cvv + (1 - a_.w) * bv[d, sl]
                nq = a_.w * cqq + (1 - a_.w) * bq[d, sl]
                dv.append(np.abs(nv - bv[d, sl]).mean())
                bv[d, sl] = nv
                bq[d, sl] = nq
                n_cells += nv.size
            assert np.isfinite(bv).all() and np.isfinite(bq).all()
            np.savez(out / f"{pn}_{sp}.npz", speed=bv.astype(np.float32), flow=bq.astype(np.float32))
            print(f"{pn} {sp}: {len(r)} blackouts, {n_cells} cells replaced, mean |dv| vs base {np.mean(dv):.2f} km/h", flush=True)
    (out / "predict_info.json").write_text(json.dumps(vars(a_), indent=1))


if __name__ == "__main__":
    main()
