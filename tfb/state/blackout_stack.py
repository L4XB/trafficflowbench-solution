"""LightGBM corrector for the U-Net inside the evaluation months' 90-minute blackouts (Task 1).

Rows: every eligible cell inside a simulated blackout (blackout_predict.py) with a finite truth. Target: truth - U-Net (speed,
and flow per lane). Features (window_features, shared with the application to the evaluation splits): the U-Net prediction and
its neighbours in time/space, the last observation before (<= T) and the first after (>= T+19) the blackout per link and its
neighbours, the time-linear interpolation, the offset inside the blackout, static link parameters and congestion context.
Fit on the non-holdout31 days, scored on holdout31's Task-1 targets.

Run: python -m tfb.state.blackout_stack --pred bo_pred_long46 [--sub 0.3] [--save bo_stack_long46]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfb.common import PROJECT, QUEUE_PANELS, load_panel
from tfb.queue.features import link_direction

HIST, BLACK = 12, 18  # same as blackout_eval.py; torch must not load next to LightGBM here (two libomp -> segfault)
REGIMES = ["R1", "R2", "R3"]
FEATS = [
    "k",
    "kr",
    "pv",
    "pq",
    "pv_up",
    "pv_dn",
    "pv_up2",
    "pv_dn2",
    "pv_tm1",
    "pv_tp1",
    "v_bef",
    "v_aft",
    "q_bef",
    "q_aft",
    "age_b",
    "age_a",
    "lin",
    "pv_minus_lin",
    "vb_up",
    "vb_dn",
    "va_up",
    "va_dn",
    "pvT",
    "pvT19",
    "v_cut",
    "v_free",
    "k_crit",
    "cap",
    "lanes",
    "on_att",
    "off_att",
    "pos",
    "tod",
    "wd",
    "ncong_b",
    "ncong_a",
    "regime",
    "panel",
    "slope_b",
    "slope_a",
    "pv_span_min",
    "pv_span_mean",
    "ncong_pv",
    "ron_t",
    "roff_t",
    "ron_d",
    "roff_d",
    "ron_w",
    "roff_w",
    "ron_wd",
    "roff_wd",
    "onset",
]


def train_conditions(pn: str) -> dict:
    """(date, T) -> 1.0 for onset / 0.0 for ongoing candidate windows of the train split."""
    w = pd.read_csv(PROJECT / "results_top3" / "queue" / "windows_train_mix.csv")
    w = w[w.panel == pn].drop_duplicates(["date", "T"])
    return {(d, int(t)): float(c == "onset") for d, t, c in zip(w.date, w["T"], w.condition)}


def ramp_link_flows(p) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-link attached ramp flow (days, T, L) for on- and off-ramps plus validity flags (numpy copy of state/data.py's)."""
    rf = np.asarray(p["ramp_flow"], dtype=np.float32)
    re = np.asarray(p["ramp_elig"])
    valid = re & np.isfinite(rf)
    rl = np.asarray(p.meta["ramps"]["ramp_link"])
    rt = np.asarray(p.meta["ramps"]["ramp_type"])
    D, T, _ = rf.shape
    L = p.n_links
    on = np.zeros((D, T, L), np.float32)
    off = np.zeros((D, T, L), np.float32)
    on_ok = np.zeros((D, T, L), bool)
    off_ok = np.zeros((D, T, L), bool)
    for i, (typ, l) in enumerate(zip(rt, rl)):
        if l < 0:
            continue
        f = np.where(valid[:, :, i], rf[:, :, i], 0.0)
        if typ.upper() in ("OR", "ON"):
            on[:, :, l] += f
            on_ok[:, :, l] |= valid[:, :, i]
        else:
            off[:, :, l] += f
            off_ok[:, :, l] |= valid[:, :, i]
    return on, off, on_ok, off_ok


def smooth_time(c: np.ndarray, passes: int) -> np.ndarray:
    """[1/4, 1/2, 1/4] smoothing along the blackout steps (axis 0), `passes` times, edges reflected."""
    for _ in range(passes):
        pad = np.concatenate([c[:1], c, c[-1:]], axis=0)
        c = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    return c


def holdout_dates() -> list[str]:
    """The holdout31 dates (config/holdout31.json)."""
    return json.loads((PROJECT / "config" / "holdout31.json").read_text())["holdout_dates"]


def last_next_obs(obs: np.ndarray):
    """Per (t, l): index of the last observed step <= t and the first observed step >= t (-1 / 288 when none)."""
    T, _L = obs.shape
    t = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(obs, t, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(obs, t, T)[::-1], axis=0)[::-1]
    return last, nxt


def window_features(
    p,
    pn: str,
    vobs: np.ndarray,
    qobs_l: np.ndarray,
    obs: np.ndarray,
    pv: np.ndarray,
    pq_l: np.ndarray,
    T: int,
    weekday: int,
    regime: str,
    ramps: tuple | None = None,
    onset: float = np.nan,
) -> np.ndarray:
    """Features (BLACK, L, n_feats) for the blackout T+1..T+18 of one day.

    vobs/qobs_l: (288, L) observations (speed, flow per lane) of the day; obs: (288, L) which of them are visible;
    pv/pq_l: (288, L) U-Net speed / flow-per-lane predictions of the day made with the blackout blank."""
    L = p.n_links
    li = np.arange(L)
    vcut = p.param("v_cut")
    vfree = p.param("free_speed_links")
    kc = p.param("k_crit")
    cap = p.param("capacity")
    lanes = p.param("lanes")
    on_a = p.param("on_attached")
    off_a = p.param("off_attached")
    dr = link_direction(pn)
    last, nxt = last_next_obs(obs)
    t0 = last[T]
    t1 = nxt[T + 1 + BLACK] if T + 1 + BLACK < 288 else np.full(L, 288)
    ok_b = t0 >= 0
    ok_a = t1 < 288
    c0, c1 = np.clip(t0, 0, 287), np.clip(t1, 0, 287)
    v_bef = np.where(ok_b, vobs[c0, li], np.nan)
    v_aft = np.where(ok_a, vobs[c1, li], np.nan)
    q_bef = np.where(ok_b, qobs_l[c0, li], np.nan)
    q_aft = np.where(ok_a, qobs_l[c1, li], np.nan)
    t0b = last[max(T - 3, 0)]
    t1a = nxt[min(T + 4 + BLACK, 287)]  # an earlier / later observation for the trends
    v_b2 = np.where((t0b >= 0) & ok_b, vobs[np.clip(t0b, 0, 287), li], np.nan)
    v_a2 = np.where((t1a < 288) & ok_a, vobs[np.clip(t1a, 0, 287), li], np.nan)
    slope_b = v_bef - v_b2
    slope_a = v_a2 - v_aft
    span = pv[T + 1 : T + 1 + BLACK]
    span_min = span.min(axis=0)
    span_mean = span.mean(axis=0)
    ker = np.ones(11)
    ncb = np.convolve(np.nan_to_num((v_bef <= vcut).astype(float)), ker, "same")
    nca = np.convolve(np.nan_to_num((v_aft <= vcut).astype(float)), ker, "same")
    up = np.clip(li - dr, 0, L - 1)
    dn = np.clip(li + dr, 0, L - 1)
    up2 = np.clip(li - 2 * dr, 0, L - 1)
    dn2 = np.clip(li + 2 * dr, 0, L - 1)
    if ramps is None:
        ron = roff = np.zeros((288, L), np.float32)
        ron_ok = roff_ok = np.zeros((288, L), bool)
    else:
        ron, roff, ron_ok, roff_ok = ramps
    k7 = np.ones(7)
    ronm = np.where(ron_ok, ron, np.nan)
    roffm = np.where(roff_ok, roff, np.nan)
    ron_w = np.apply_along_axis(lambda x: np.convolve(x, k7, "same"), 1, ron)
    roff_w = np.apply_along_axis(lambda x: np.convolve(x, k7, "same"), 1, roff)
    out = np.empty((BLACK, L, len(FEATS)), np.float32)
    pid = QUEUE_PANELS.index(pn)
    rid = REGIMES.index(regime)
    for j, t in enumerate(range(T + 1, T + 1 + BLACK)):
        k = t - T
        lin = np.where(ok_b & ok_a, v_bef + (v_aft - v_bef) * (t - t0) / np.maximum(t1 - t0, 1), np.nan)
        v = pv[t]
        out[j] = np.stack(
            [
                np.full(L, k),
                np.full(L, BLACK + 1 - k),
                v,
                pq_l[t],
                v[up],
                v[dn],
                v[up2],
                v[dn2],
                pv[t - 1],
                pv[min(t + 1, 287)],
                v_bef,
                v_aft,
                q_bef,
                q_aft,
                T - t0,
                t1 - (T + 1 + BLACK),
                lin,
                v - lin,
                v_bef[up],
                v_bef[dn],
                v_aft[up],
                v_aft[dn],
                pv[T],
                pv[min(T + 1 + BLACK, 287)],
                vcut,
                vfree,
                kc,
                cap,
                lanes,
                on_a,
                off_a,
                li / max(L - 1, 1),
                np.full(L, T),
                np.full(L, weekday),
                ncb,
                nca,
                np.full(L, rid),
                np.full(L, pid),
                slope_b,
                slope_a,
                span_min,
                span_mean,
                np.convolve((v <= vcut).astype(float), ker, "same"),
                ronm[t],
                roffm[t],
                ronm[t] - ronm[T],
                roffm[t] - roffm[T],
                ron_w[t],
                roff_w[t],
                ron_w[t] - ron_w[T],
                roff_w[t] - roff_w[T],
                np.full(L, onset),
            ],
            axis=1,
        )
    return out


def panel_rows(pn: str, pred_dir: Path, rng: np.random.Generator, sub: float):
    """Rows for one train panel's blackout cells in <pred_dir>/<pn>.npz: eligible, finite truth, a random share `sub` off holdout31.

    Returns X (n, len(FEATS)), y (n, 2) truth - U-Net in speed and flow per lane, info (n, 5): day, step, link, holdout31, Task-1 target.
    """
    p = load_panel(pn, "train")
    z = np.load(pred_dir / f"{pn}.npz")
    wins = z["wins"]
    pvB = z["pvB"].astype(np.float32)
    pqB = z["pqB"].astype(np.float32)
    tv = np.asarray(p["speed"], dtype=np.float32)
    tq = np.asarray(p["flow"], dtype=np.float32)
    el = np.asarray(p["elig"])
    tg = np.asarray(p["target"]).copy()
    lanes = p.param("lanes").astype(np.float32)
    h31 = set(holdout_dates())
    is_h = np.array([d in h31 for d in p.dates])
    reg = np.asarray(p.regimes)
    wd = np.array([pd.Timestamp(d).weekday() for d in p.dates])
    black = np.zeros((len(p.dates), 288), bool)
    for d, T in wins:
        black[d, T + 1 : T + 1 + BLACK] = True
        tg[d, T - HIST : T] = False
    qpl = tq / lanes[None, None, :]
    pql = pqB / lanes[None, None, :]
    ron, roff, ron_ok, roff_ok = ramp_link_flows(p)
    conds = train_conditions(pn)
    rows, ys, info = [], [], []
    for d, T in wins:
        obs = np.isfinite(tv[d]) & ~tg[d] & ~black[d][:, None]
        F = window_features(
            p,
            pn,
            tv[d],
            qpl[d],
            obs,
            pvB[d],
            pql[d],
            int(T),
            int(wd[d]),
            str(reg[d]),
            (ron[d], roff[d], ron_ok[d], roff_ok[d]),
            conds.get((p.dates[d], int(T)), np.nan),
        )
        sl = slice(T + 1, T + 1 + BLACK)
        m = el[d, sl] & np.isfinite(tv[d, sl])
        keep = m if (is_h[d] or sub >= 1.0) else m & (rng.random(m.shape) < sub)
        if not keep.any():
            continue
        rows.append(F[keep])
        ys.append(np.stack([tv[d, sl][keep] - pvB[d, sl][keep], qpl[d, sl][keep] - pql[d, sl][keep]], axis=1).astype(np.float32))
        tt, ll = np.nonzero(keep)
        info.append(
            np.stack([np.full(len(tt), d), tt + T + 1, ll, np.full(len(tt), int(is_h[d])), tg[d, sl][keep].astype(int)], axis=1).astype(
                np.int32
            )
        )
    return np.concatenate(rows), np.concatenate(ys), np.concatenate(info)


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--sub", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--all-days", action="store_true", help="fit on every day including holdout31 (final corrector; no score)")
    ap.add_argument("--save", default="", help="save the two boosters under results_top3/state/<save>/")
    ap.add_argument("--drop", nargs="*", default=[], help="feature names left out (e.g. location identity: pos panel)")
    a_ = ap.parse_args()
    rng = np.random.default_rng(42)
    X, Y, I = [], [], []
    for pi, pred in enumerate(a_.pred):
        pred_dir = PROJECT / "results_top3" / "state" / pred
        for pn in QUEUE_PANELS:
            x, y, info = panel_rows(pn, pred_dir, rng, a_.sub)
            if pi > 0:
                info[:, 4] = 0  # extra placements train only; the score stays on the first placement
            X.append(x)
            Y.append(y)
            I.append(info)
            print(f"{pred} {pn}: {len(x)} rows ({info[:, 3].mean():.1%} holdout31)", flush=True)
    X = np.ascontiguousarray(np.concatenate(X))
    Y = np.concatenate(Y)
    I = np.concatenate(I)
    P = X[:, FEATS.index("panel")].astype(int)
    use = [f for f in FEATS if f not in a_.drop]
    X = np.ascontiguousarray(X[:, [FEATS.index(f) for f in use]])
    cats = [c for c in ("panel", "regime") if c in use]
    kcol = use.index("k")
    tr = np.ones(len(X), bool) if a_.all_days else I[:, 3] == 0
    te = (I[:, 3] == 1) & (I[:, 4] == 1)
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": a_.leaves,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 6,
        "seed": 42,
    }
    rm = lambda e: float(np.sqrt(np.mean(e.astype(np.float64) ** 2)))
    res, boosters = {}, []
    for j, nm in enumerate(["speed", "flow"]):
        dtr = lgb.Dataset(X[tr], Y[tr, j], feature_name=use, categorical_feature=cats, free_raw_data=False)
        b = lgb.train(params, dtr, num_boost_round=a_.rounds)
        boosters.append(b)
        if not a_.all_days:
            corr = b.predict(X[te], num_threads=6)
            base = rm(Y[te, j])
            new = rm(Y[te, j] - corr)
            res[nm] = (base, new)
            print(f"{nm}: holdout31 blackout targets rmse U-Net {base:.3f} -> corrected {new:.3f} (n {te.sum()})", flush=True)
            for kk in [1, 6, 12, 18]:
                mk = te & (X[:, kcol] == kk)
                print(f"   k={kk:2d}: {rm(Y[mk, j]):.2f} -> {rm(Y[mk, j] - b.predict(X[mk], num_threads=6)):.2f}")
            per_p = {
                QUEUE_PANELS[i]: (rm(Y[te & (P == i), j]), rm(Y[te & (P == i), j] - b.predict(X[te & (P == i)], num_threads=6)))
                for i in range(len(QUEUE_PANELS))
            }
            print("   per panel:", {k: f"{v[0]:.2f}->{v[1]:.2f}" for k, v in per_p.items()}, flush=True)
    imp = sorted(zip(use, boosters[0].feature_importance("gain")), key=lambda x: -x[1])[:12]
    print("speed gain importance:", [(n, f"{g:.3g}") for n, g in imp])
    if a_.save:
        out = PROJECT / "results_top3" / "state" / a_.save
        out.mkdir(parents=True, exist_ok=True)
        boosters[0].save_model(str(out / "speed.txt"))
        boosters[1].save_model(str(out / "flow.txt"))
        (out / "meta.json").write_text(json.dumps({"feats": use, "pred": a_.pred, "res": res, "all_days": a_.all_days}, indent=1))


if __name__ == "__main__":
    main()
