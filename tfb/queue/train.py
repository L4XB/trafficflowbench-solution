"""Train and evaluate the onset and ongoing queue models (forward protocol).

Run: python -m tfb.queue.train [--protocol forward|lomo]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.common import PROJECT, QUEUE_PANELS, STEPS
from tfb.queue.data import HOR
from tfb.queue.evaluation import TEST_MONTHS, PanelQ, iou, persist_last_elig, windows_path
from tfb.queue.features import (
    Climatology,
    _dir,
    abs_speed_feats,
    balance_feats,
    corridor_feats,
    corridor_feats2,
    corridor_feats3,
    dayctx_feats,
    edge_feats,
    horizon_clim,
    inmonth_day_feats,
    inmonth_demand_feats,
    inmonth_feats,
    inmonth_horizon,
    inmonth_onset_feats,
    link_feats,
    origin_feats,
    remcap_feats,
    shock_feats,
    static_feats,
    trend_feats,
)
from tfb.scoring import queue_aggregate

OUT = PROJECT / "results_top3" / "queue"
PANEL_ID = {pn: i for i, pn in enumerate(QUEUE_PANELS)}
RNG = np.random.default_rng(42)


def build_rows(
    windows: pd.DataFrame,
    pqs: dict,
    clim: Climatology,
    statics: dict,
    cond: str,
    cache: Path | None = None,
    feats: str = "v1",
    drop_rate=None,
):
    """Feature matrix and labels. onset: one row per link (label at T+30). ongoing: one row per (h, link).

    feats='v2' adds the origin-step observation (with simulated masking at `drop_rate`, or the day's
    regime rate when drop_rate='regime') and the history trend features.
    """
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["keys"], list(z["names"])
    Xs, ys, keys = [], [], []
    rng = np.random.default_rng(1234)
    rates = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
    if os.environ.get("QUEUE_NODROP") == "1":  # filled histories: the origin step is complete at evaluation too
        drop_rate = None
    clim_all = clim
    for r in windows[windows.condition == cond].itertuples():
        pq = pqs[r.panel]
        L = pq.p.n_links
        clim = clim_all[r.month] if isinstance(clim_all, dict) else clim_all  # leave-own-month-out climatology in training
        restore = pq.patch(r.d, r.T)  # causal fill of this window's history (no-op otherwise)
        X, names = link_feats(pq, r.d, r.T, clim, *statics[r.panel])
        if feats in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"):
            if pq.split == "train":
                dr = rates[rng.choice(list(rates))] if drop_rate == "regime" else drop_rate
            else:
                dr = None
            Xo, no = origin_feats(pq, r.d, r.T, rng, dr)
            Xt, nt = trend_feats(pq, r.d, r.T)
            X = np.concatenate([X, Xo, Xt], 1)
            names = names + no + nt
            if feats in ("v3", "v15"):
                Xd, nd = dayctx_feats(pq, r.d, r.T, rng, dr)
                X = np.concatenate([X, Xd], 1)
                names = names + nd
            if feats == "v4":
                Xb, nb = balance_feats(pq, r.d, r.T)
                X = np.concatenate([X, Xb], 1)
                names = names + nb
            if feats == "v5":
                Xa, na = abs_speed_feats(pq, r.d, r.T)
                X = np.concatenate([X, Xa], 1)
                names = names + na
            if feats in ("v6", "v8", "v9", "v10", "v11", "v12", "v13", "v15", "v16", "v17"):
                Xi, ni = inmonth_feats(pq, r.d, r.T, rng, dr, cond)
                X = np.concatenate([X, Xi], 1)
                names = names + ni
            if feats in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"):
                Xc, nc_ = corridor_feats(pq, r.d, r.T)
                X = np.concatenate([X, Xc], 1)
                names = names + nc_
            if feats in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"):
                Xc2, nc2 = corridor_feats2(pq, r.d, r.T)
                X = np.concatenate([X, Xc2], 1)
                names = names + nc2
            if feats == "v11":
                Xsh, nsh = shock_feats(pq, r.d, r.T)
                X = np.concatenate([X, Xsh], 1)
                names = names + nsh
            if feats == "v12":
                Xb3, nb3 = corridor_feats3(pq, r.d, r.T)
                X = np.concatenate([X, Xb3], 1)
                names = names + nb3
            if feats == "v13":
                Xeg, neg = edge_feats(pq, r.d, r.T)
                X = np.concatenate([X, Xeg], 1)
                names = names + neg
            if feats in ("v16", "v17"):
                Xmd, nmd = inmonth_day_feats(pq, r.d, r.T)
                X = np.concatenate([X, Xmd], 1)
                names = names + nmd
            if feats == "v17":
                Xmq, nmq = inmonth_demand_feats(pq, r.d, r.T, cond)
                X = np.concatenate([X, Xmq], 1)
                names = names + nmq
            if feats == "v8":
                Xo8, no8 = inmonth_onset_feats(pq, r.d, r.T, rng, dr, cond)
                X = np.concatenate([X, Xo8], 1)
                names = names + no8
            if feats == "v7":
                Xr, nr = remcap_feats(pq, r.d, r.T)
                X = np.concatenate([X, Xr], 1)
                names = names + nr
        restore()
        fc = pq.forecast(r.d, r.T)
        if fc is not None:
            X, names = np.concatenate([X, fc_link_feats(fc, _dir(pq))], 1), names + FC_LINK_NAMES
            if FC_FEATS() == 2:
                fq, fk = pq.forecast_qk(r.d, r.T)
                X, names = np.concatenate([X, fc_link_feats2(fc, fq, fk, _dir(pq))], 1), names + FC2_LINK_NAMES
            if FC_FEATS() == 3:
                fp = pq.forecast_p(r.d, r.T)
                X, names = np.concatenate([X, fc_link_feats3(fp, _dir(pq))], 1), names + FC3_LINK_NAMES
        pid = np.full((L, 1), PANEL_ID[r.panel], float)
        if cond == "onset":
            y = np.asarray(pq.truth[r.d, r.T + HOR]).astype(np.int8)
            Xs.append(np.concatenate([X, pid], 1).astype(np.float32))
            ys.append(y)
            keys.append(np.column_stack([np.full(L, r.Index), np.arange(L), np.zeros(L)]))
        else:
            hc = horizon_clim(pq, r.T, clim)  # (HOR, L)
            ih = inmonth_horizon(pq, r.d, r.T, rng, dr) if INMH_ON() else None
            for h in range(1, HOR + 1):
                extra = np.column_stack([np.full(L, float(h)), hc[h - 1], pid[:, 0]])
                if ih is not None:
                    extra = np.column_stack([ih[h - 1], extra])
                if fc is not None:
                    extra = np.column_stack(
                        [fc_h_feats(fc, h, _dir(pq))]
                        + ([fc_h_feats2(fc, fq, fk, h)] if FC_FEATS() == 2 else [])
                        + ([fc_h_feats3(fp, h, _dir(pq))] if FC_FEATS() == 3 else [])
                        + [extra]
                    )
                Xs.append(np.concatenate([X, extra], 1).astype(np.float32))
                ys.append(np.asarray(pq.truth[r.d, r.T + h]).astype(np.int8))
                keys.append(np.column_stack([np.full(L, r.Index), np.arange(L), np.full(L, h)]))
    names = names + (
        ["panel"]
        if cond == "onset"
        else ((FC_H_NAMES + (FC2_H_NAMES if FC_FEATS() == 2 else []) + (FC3_H_NAMES if FC_FEATS() == 3 else [])) if FC_ON() else [])
        + (["inm_hh"] if INMH_ON() else [])
        + ["h", "clim_hh", "panel"]
    )
    X, y, keys = np.concatenate(Xs), np.concatenate(ys), np.concatenate(keys).astype(int)
    if cache is not None:
        np.savez(cache, X=X, y=y, keys=keys, names=np.array(names))
    return X, y, keys, names


FC_LINK_NAMES = ["fc_r6", "fc_rmin", "fc_r6_up", "fc_r6_dn", "fc_rmin_nb", "fc_r3"]
FC_H_NAMES = ["fc_rh", "fc_rh_up", "fc_rh_dn"]


def FC_ON() -> bool:
    """U-Net forecast features on: QUEUE_FC is set (suffix of the per-window forecast files, see PanelQ)."""
    return bool(os.environ.get("QUEUE_FC"))


def FC_FEATS() -> int:
    """Forecast feature set: 1 = speed ratios only, 2 = + flow / density forecasts and temporal neighbours (QUEUE_FC_FEATS)."""
    return int(os.environ.get("QUEUE_FC_FEATS", "1")) if FC_ON() else 0


def DIR_TAG() -> str:
    """Run-tag suffix of the neighbour and in-month options: _dir (QUEUE_DIRFIX=1), _inmh (QUEUE_INMH=1), _roll[days] (QUEUE_INM_ROLL=1)."""
    return (
        ("_dir" if os.environ.get("QUEUE_DIRFIX") == "1" else "")
        + ("_inmh" if INMH_ON() else "")
        + (
            ("_roll" + ("" if os.environ.get("QUEUE_INM_ROLL_DAYS", "16") == "16" else os.environ["QUEUE_INM_ROLL_DAYS"]))
            if os.environ.get("QUEUE_INM_ROLL") == "1"
            else ""
        )
    )


def INMH_ON() -> bool:
    """Ongoing per-horizon in-month recurrence feature inm_hh (QUEUE_INMH=1)."""
    return os.environ.get("QUEUE_INMH") == "1"


def FC_TAG() -> str:
    """Run-tag suffix of the forecast features: _fc<QUEUE_FC> plus _ff2 / _ff3 for feature sets 2 / 3; empty when they are off."""
    return (f"_fc{os.environ['QUEUE_FC']}" + {1: "", 2: "_ff2", 3: "_ff3"}[FC_FEATS()]) if FC_ON() else ""


FC2_LINK_NAMES = ["fc_q6", "fc_qmax", "fc_k6", "fc_kmax", "fc_rmean", "fc_q6_dn"]
FC2_H_NAMES = ["fc_qh", "fc_kh", "fc_rh_prev", "fc_rh_next"]


def fc_link_feats2(fc: np.ndarray, fq: np.ndarray, fk: np.ndarray, dr: int = 1) -> np.ndarray:
    """fc: speed / v_cut, fq: flow / capacity, fk: density / k_crit, each (HOR, L) -> per-link features."""
    return np.column_stack([fq[-1], fq.max(0), fk[-1], fk.max(0), fc.mean(0), _nb(fq[-1], -dr)])


FC3_LINK_NAMES = ["fc_p6", "fc_pmax", "fc_pmean", "fc_p6_up", "fc_p6_dn"]
FC3_H_NAMES = ["fc_ph", "fc_ph_up", "fc_ph_dn"]


def fc_link_feats3(fp: np.ndarray, dr: int = 1) -> np.ndarray:
    """fp: (HOR, L) forecast queue probability from the queue-head forecaster -> per-link features."""
    return np.column_stack([fp[-1], fp.max(0), fp.mean(0), _nb(fp[-1], dr), _nb(fp[-1], -dr)])


def fc_h_feats3(fp: np.ndarray, h: int, dr: int = 1) -> np.ndarray:
    """fp: (HOR, L) forecast queue probability -> (L, 3) its step h on the link and on its two neighbours (dr: neighbour direction)."""
    ph = fp[h - 1]
    return np.column_stack([ph, _nb(ph, dr), _nb(ph, -dr)])


def fc_h_feats2(fc: np.ndarray, fq: np.ndarray, fk: np.ndarray, h: int) -> np.ndarray:
    """(L, 4) at horizon step h: forecast flow / capacity, density / k_crit, and speed / v_cut one step earlier and later (clamped)."""
    prev = fc[max(h - 2, 0)]
    nxt = fc[min(h, fc.shape[0] - 1)]
    return np.column_stack([fq[h - 1], fk[h - 1], prev, nxt])


def _nb(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    if k > 0:
        out[k:] = a[:-k]
    else:
        out[:k] = a[-k:]
    return out


def fc_link_feats(fc: np.ndarray, dr: int = 1) -> np.ndarray:
    """fc: (HOR, L) forecast speed / v_cut -> per-link features (dr: neighbour direction, see tfb.queue.features._dir)."""
    r6 = fc[-1]
    rmin = fc.min(0)
    return np.column_stack([r6, rmin, _nb(r6, dr), _nb(r6, -dr), np.fmin(_nb(rmin, 1), _nb(rmin, -1)), fc[2]])


def fc_h_feats(fc: np.ndarray, h: int, dr: int = 1) -> np.ndarray:
    """fc: (HOR, L) forecast speed / v_cut -> (L, 3) its step h on the link and on its two neighbours (dr: neighbour direction)."""
    rh = fc[h - 1]
    return np.column_stack([rh, _nb(rh, dr), _nb(rh, -dr)])


def expected_iou_prefix(p: np.ndarray, order: np.ndarray, max_k: int, n_samples: int = 256) -> tuple[int, float]:
    """Best prefix length of `order` under independent Bernoulli(p); returns (k, E[IoU])."""
    ps = p[order]
    samp = RNG.random((n_samples, len(ps))) < ps[None, :]  # (S, n) sampled truths (over the ordered cells)
    tot_true = samp.sum(1)  # |Y| per sample (cells outside `order` assumed 0)
    cum_inter = np.cumsum(samp, 1)  # |S_k ∩ Y|
    best_k, best = 1, -1.0
    for k in range(1, min(max_k, len(ps)) + 1):
        inter = cum_inter[:, k - 1]
        union = k + tot_true - inter
        e = np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
        if e > best:
            best, best_k = e, k
    return best_k, float(best)


def decode(prob: np.ndarray, mode: str, k: int = 3, thr: float = 0.5, max_k: int = 40) -> np.ndarray:
    """prob: (n_cells,) -> boolean selection."""
    sel = np.zeros(prob.shape, dtype=bool)
    if mode == "topk":
        order = np.argsort(-prob)[:k]
        sel[order[prob[order] > 0]] = True
    elif mode == "thr":
        sel = prob >= thr
        if not sel.any():
            sel[np.argmax(prob)] = True
    elif mode == "eiou":
        order = np.argsort(-prob)
        cand = order[prob[order] > 0.003]
        if len(cand) == 0:
            cand = order[:1]
        kk, _ = expected_iou_prefix(prob, cand, max_k)
        sel[cand[:kk]] = True
    return sel


def decode_block(p: np.ndarray, max_len: int = 12, n_samples: int = 256) -> np.ndarray:
    """Best contiguous link block under independent Bernoulli(p) by Monte-Carlo expected IoU."""
    L = len(p)
    samp = RNG.random((n_samples, L)) < p[None, :]
    tot = samp.sum(1)
    cs = np.concatenate([np.zeros((n_samples, 1), int), np.cumsum(samp, 1)], 1)
    cand = np.where(p > 0.03)[0]
    best, best_ab = -1.0, (int(np.argmax(p)), int(np.argmax(p)))
    for a in cand:
        for b in range(a, min(L, a + max_len)):
            if p[b] < 0.03 and b > a and p[a : b + 1].max() < 0.03:
                break
            inter = cs[:, b + 1] - cs[:, a]
            k = b - a + 1
            union = k + tot - inter
            e = np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
            if e > best:
                best, best_ab = e, (a, b)
    sel = np.zeros(L, dtype=bool)
    sel[best_ab[0] : best_ab[1] + 1] = True
    return sel


def fit_model(X, y, names, params: dict, rounds: int, weight=None) -> lgb.Booster:
    """LightGBM booster trained `rounds` rounds on (X, y, weight), with the panel column (if any) categorical."""
    ds = lgb.Dataset(
        X, y, weight=weight, feature_name=names, categorical_feature=["panel"] if "panel" in names else [], free_raw_data=False
    )
    return lgb.train(params, ds, num_boost_round=rounds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="forward")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--feats", default="v1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--onset-block", action="store_true", help="also decode onset as the best contiguous block")
    ap.add_argument("--truth", default="raw", help="raw (filled observations) or dn (U-Net denoised)")
    ap.add_argument("--windows-tag", default="", help="suffix of an alternative windows file, e.g. _dense -> windows_train_dn_dense.csv")
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--min-leaf", type=int, default=100)
    ap.add_argument("--ongoing-rounds", type=int, default=0, help="rounds for the ongoing model (0 = same as --rounds)")
    ap.add_argument(
        "--tilt", type=float, default=0.0, help="ongoing training weight exp(-tilt * window persistence IoU) (emphasis on dynamic windows)"
    )
    ap.add_argument("--noclim", action="store_true", help="drop the climatology columns (clim*, onset_freq*, onset_rate*) from both models")
    ap.add_argument("--seeds", type=int, default=1, help="average this many seeds per model")
    a = ap.parse_args()
    t0 = time.time()
    w = pd.read_csv(str(windows_path(a.truth)).replace(".csv", a.windows_tag + ".csv"))
    pqs = {pn: PanelQ(pn, truth=a.truth) for pn in QUEUE_PANELS}
    statics = {pn: static_feats(pqs[pn]) for pn in QUEUE_PANELS}
    months = sorted(w.month.unique())
    folds = (
        [(tuple(m for m in months if m not in TEST_MONTHS), TEST_MONTHS)]
        if a.protocol == "forward"
        else [(tuple(m for m in months if m != tm), (tm,)) for tm in months]
    )
    all_rows = []
    for fit_m, test_m in folds:
        fit_days = {pn: np.where(np.isin(pqs[pn].months, fit_m))[0] for pn in QUEUE_PANELS}
        clim = Climatology(pqs, fit_days, onset_windows=w[w.condition == "onset"])
        fit_w = w[w.month.isin(fit_m)]
        test_w = w[w.month.isin(test_m)]
        clim_fit = clim
        if os.environ.get("QUEUE_LOMC") in (
            "1",
            "onset",
            "ongoing",
        ):  # training features from a climatology that excludes the window's own month
            clim_fit = {
                m: Climatology(
                    pqs,
                    {pn: np.where(np.isin(pqs[pn].months, [x for x in fit_m if x != m]))[0] for pn in QUEUE_PANELS},
                    onset_windows=w[w.condition == "onset"],
                )
                for m in fit_m
            }
            print(f"  leave-own-month-out climatologies built for {len(clim_fit)} months ({time.time() - t0:.0f}s)", flush=True)
        print(f"fold test={test_m} fit windows {len(fit_w)} test windows {len(test_w)}  ({time.time() - t0:.0f}s)", flush=True)
        params = {
            "objective": "binary",
            "learning_rate": a.lr,
            "num_leaves": a.leaves,
            "min_data_in_leaf": a.min_leaf,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "verbose": -1,
            "seed": 42,
            "num_threads": 10,
        }
        # ---------------- onset
        tag = (
            "-".join(test_m)
            + (f"_{a.feats}" if a.feats != "v1" else "")
            + ("" if a.truth == "raw" else f"_{a.truth}")
            + a.windows_tag
            + (f"_cap{os.environ['INMONTH_CAP']}" if os.environ.get("INMONTH_CAP") else "")
            + (
                "_fill" + os.environ["QUEUE_FILL"]
                if os.environ.get("QUEUE_FILL") not in (None, "", "1")
                else ("_fill" if os.environ.get("QUEUE_FILL") == "1" else "")
            )
            + ("_nodrop" if os.environ.get("QUEUE_NODROP") == "1" else "")
            + (f"_ft{os.environ['QUEUE_FILL_TRAIN']}" if os.environ.get("QUEUE_FILL_TRAIN") else "")
            + (f"_lomc{os.environ['QUEUE_LOMC']}" if os.environ.get("QUEUE_LOMC") in ("1", "onset", "ongoing") else "")
            + FC_TAG()
            + DIR_TAG()
        )
        X, y, _keys, names = build_rows(
            fit_w,
            pqs,
            clim_fit if os.environ.get("QUEUE_LOMC") in ("1", "onset") else clim,
            statics,
            "onset",
            OUT / f"feat_onset_fit_{tag}.npz",
            a.feats,
            "regime",
        )
        print(f"  onset rows {X.shape} positives {y.sum()} ({time.time() - t0:.0f}s)", flush=True)
        Xt, _yt, kt, _ = build_rows(test_w, pqs, clim, statics, "onset", OUT / f"feat_onset_test_{tag}.npz", a.feats, "regime")
        if a.noclim:
            keep = [i for i, n in enumerate(names) if not n.startswith(("clim", "onset_freq", "onset_rate"))]
            X, Xt, names = X[:, keep], Xt[:, keep], [names[i] for i in keep]
        m_on = [fit_model(X, y, names, {**params, "seed": 42 + i}, a.rounds) for i in range(a.seeds)]
        pt = np.mean([m.predict(Xt) for m in m_on], 0)
        imp = pd.Series(m_on[0].feature_importance("gain"), index=names).sort_values(ascending=False)
        print("  onset top features:", imp.head(15).round(0).to_dict(), flush=True)
        # ---------------- ongoing
        X2, y2, k2, names2 = build_rows(
            fit_w,
            pqs,
            clim_fit if os.environ.get("QUEUE_LOMC") in ("1", "ongoing") else clim,
            statics,
            "ongoing",
            OUT / f"feat_ongoing_fit_{tag}.npz",
            a.feats,
            "regime",
        )
        print(f"  ongoing rows {X2.shape} positives {y2.sum()} ({time.time() - t0:.0f}s)", flush=True)
        Xt2, _yt2, kt2, _ = build_rows(test_w, pqs, clim, statics, "ongoing", OUT / f"feat_ongoing_test_{tag}.npz", a.feats, "regime")
        if a.noclim:
            keep2 = [i for i, n in enumerate(names2) if not n.startswith(("clim", "onset_freq", "onset_rate"))]
            X2, Xt2, names2 = X2[:, keep2], Xt2[:, keep2], [names2[i] for i in keep2]
        wt2 = np.exp(-a.tilt * w.loc[k2[:, 0], "piou"].to_numpy(float)) if a.tilt > 0 else None
        m_og = [fit_model(X2, y2, names2, {**params, "seed": 42 + i}, a.ongoing_rounds or a.rounds, weight=wt2) for i in range(a.seeds)]
        pt2 = np.mean([m.predict(Xt2) for m in m_og], 0)
        imp2 = pd.Series(m_og[0].feature_importance("gain"), index=names2).sort_values(ascending=False)
        print("  ongoing top features:", imp2.head(15).round(0).to_dict(), flush=True)
        # ---------------- decode & score per window
        rows = []
        for widx in np.unique(kt[:, 0]):
            r = w.loc[widx]
            pq = pqs[r["panel"]]
            L = pq.p.n_links
            rd, rT = int(r["d"]), int(r["T"])
            sel = kt[:, 0] == widx
            prob = pt[sel]
            links = kt[sel, 1]
            tr = pq.truth_window(rd, rT)
            for mode, kw in (
                ("topk", {"k": 2}),
                ("topk", {"k": 3}),
                ("topk", {"k": 4}),
                ("thr", {"thr": 0.5}),
                ("thr", {"thr": 0.35}),
                ("thr", {"thr": 0.25}),
                ("eiou", {}),
            ):
                pred = np.zeros((HOR, L), dtype=bool)
                s = decode(prob, mode, **kw)
                pred[-1, links[s]] = True
                rows.append(
                    {
                        "model": f"onset_{mode}{kw.get('k', kw.get('thr', ''))}",
                        "panel": r["panel"],
                        "condition": "onset",
                        "d": rd,
                        "T": rT,
                        "iou": iou(pred, tr),
                    }
                )
            if a.onset_block:
                pred = np.zeros((HOR, L), dtype=bool)
                pl = np.zeros(L)
                pl[links] = prob
                pred[-1] = decode_block(pl)
                rows.append({"model": "onset_block", "panel": r["panel"], "condition": "onset", "d": rd, "T": rT, "iou": iou(pred, tr)})
            # climatology-only reference
            pred = np.zeros((HOR, L), dtype=bool)
            f = clim.freq[r["panel"]][(rT + HOR) % STEPS]
            top = np.argsort(-f)[:3]
            pred[-1, top[f[top] > 0]] = True
            rows.append({"model": "onset_clim_top3", "panel": r["panel"], "condition": "onset", "d": rd, "T": rT, "iou": iou(pred, tr)})
        for widx in np.unique(kt2[:, 0]):
            r = w.loc[widx]
            pq = pqs[r["panel"]]
            L = pq.p.n_links
            rd, rT = int(r["d"]), int(r["T"])
            sel = kt2[:, 0] == widx
            prob = pt2[sel]
            links = kt2[sel, 1]
            hs = kt2[sel, 2]
            tr = pq.truth_window(rd, rT)
            for mode, kw in (("thr", {"thr": 0.5}), ("thr", {"thr": 0.4}), ("thr", {"thr": 0.3}), ("eiou", {"max_k": 2000})):
                pred = np.zeros((HOR, L), dtype=bool)
                s = decode(prob, mode, **kw)
                pred[hs[s] - 1, links[s]] = True
                rows.append(
                    {
                        "model": f"ongoing_{mode}{kw.get('thr', '')}",
                        "panel": r["panel"],
                        "condition": "ongoing",
                        "d": rd,
                        "T": rT,
                        "iou": iou(pred, tr),
                    }
                )
            for name, sched in (
                ("ongoing_hdown", [0.55, 0.5, 0.5, 0.45, 0.45, 0.4]),
                ("ongoing_hup", [0.4, 0.45, 0.45, 0.5, 0.5, 0.55]),
                ("ongoing_hdown2", [0.6, 0.55, 0.5, 0.45, 0.4, 0.35]),
            ):
                thr_h = np.array(sched)[hs - 1]
                pred = np.zeros((HOR, L), dtype=bool)
                s = prob >= thr_h
                pred[hs[s] - 1, links[s]] = True
                rows.append({"model": name, "panel": r["panel"], "condition": "ongoing", "d": rd, "T": rT, "iou": iou(pred, tr)})
            rows.append(
                {
                    "model": "ongoing_persist",
                    "panel": r["panel"],
                    "condition": "ongoing",
                    "d": rd,
                    "T": rT,
                    "iou": iou(persist_last_elig(pq, rd, rT, "ongoing"), tr),
                }
            )
        res = pd.DataFrame(rows)
        res["fold"] = str(test_m)
        all_rows.append(res)
        res = res.merge(w[["panel", "d", "T", "piou"]], on=["panel", "d", "T"], how="left")
        for name, g in res.groupby("model", sort=False):
            pc = g.groupby("panel").iou.mean()
            tw = np.exp(-1.462 * g.piou.fillna(0).to_numpy(float))
            tilted = float(np.average(g.iou, weights=tw)) if name.startswith("ongoing") else float("nan")
            print(
                f"  {name:22s} mean {g.iou.mean():.4f} tilt1.46 {tilted:.4f} | " + " ".join(f"{pn[3:]}:{v:.3f}" for pn, v in pc.items()),
                flush=True,
            )
    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(OUT / f"model_{a.protocol}{a.tag}.csv", index=False)
    # combined S_queue for the best pairs
    print("\n=== official aggregation on test windows ===")
    for on_name in ["onset_clim_top3", "onset_topk3", "onset_eiou", "onset_thr0.35", "onset_thr0.25"] + (
        ["onset_block"] if a.onset_block else []
    ):
        for og_name in ["ongoing_persist", "ongoing_thr0.4", "ongoing_eiou"]:
            g = res[res.model.isin([on_name, og_name])]
            agg = queue_aggregate(g)
            print(
                f"{on_name:16s} + {og_name:16s} S_queue {agg['S_queue']:.4f}  onset {agg['per_condition']['onset']:.4f} ongoing {agg['per_condition']['ongoing']:.4f}"
            )


if __name__ == "__main__":
    main()
