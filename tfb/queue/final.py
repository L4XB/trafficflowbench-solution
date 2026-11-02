"""Final fit on all train months and inference for the validation/private windows.

Run: python -m tfb.queue.final --name v1 [--onset-mode thr --onset-thr 0.35] [--ongoing-mode eiou]
Writes results_top3/queue/final_<name>/{onset.txt, ongoing.txt, pred.csv, meta.json}
"""

from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from tfb.common import QUEUE_PANELS
from tfb.queue.data import HOR, released_windows
from tfb.queue.evaluation import PanelQ, windows_path
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
from tfb.queue.train import (
    DIR_TAG,
    FC_FEATS,
    FC_TAG,
    INMH_ON,
    OUT,
    PANEL_ID,
    build_rows,
    decode,
    fc_h_feats,
    fc_h_feats2,
    fc_h_feats3,
    fc_link_feats,
    fc_link_feats2,
    fc_link_feats3,
    fit_model,
)


def noclim_idx(names):
    """Columns of the no-climatology model; QUEUE_ROBUST=1 also drops the link-identity columns pos and panel, so the model cannot
    memorise train bottleneck locations (the evaluation months have recurrent bottlenecks the train months never had)."""
    robust = os.environ.get("QUEUE_ROBUST", "0")
    static_id = (
        "lanes",
        "cap_lane",
        "vf",
        "length",
        "kjam_lane",
        "cap_ratio_up",
        "cap_ratio_dn",
        "on_att",
        "off_att",
        "on_att_up1",
        "on_att_dn1",
    )
    return [
        i
        for i, n in enumerate(names)
        if not (
            n.startswith(("clim", "onset_freq", "onset_rate"))
            or (robust in ("1", "3") and n in ("pos", "panel"))
            or (robust == "3" and n in static_id)
        )
    ]


def iso(date: str, slot: int) -> str:
    """ISO timestamp 'YYYY-MM-DDTHH:MM:00Z' of a 5-minute slot on `date` (inverse of tfb.common.t_index)."""
    return f"{date}T{slot // 12:02d}:{(slot % 12) * 5:02d}:00Z"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--onset-mode", default="thr")
    ap.add_argument("--onset-thr", type=float, default=0.35)
    ap.add_argument("--ongoing-mode", default="eiou")
    ap.add_argument("--ongoing-thr", type=float, default=0.4)
    ap.add_argument("--splits", nargs="*", default=["validation", "private"])
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--feats", default="v1")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--truth", default="raw")
    ap.add_argument("--ongoing-leaves", type=int, default=63)
    ap.add_argument(
        "--onset-hedge", action="store_true", help="also fit onset on raw labels over the same windows and take the geometric mean"
    )
    ap.add_argument("--noclim-w-onset", type=float, default=0.0, help="weight of the no-climatology onset model in the final probability")
    ap.add_argument("--dump-probs", action="store_true", help="save the per-window probability vectors to probs.npz")
    ap.add_argument(
        "--noclim-w-ongoing", type=float, default=0.0, help="weight of the no-climatology ongoing model in the final probability"
    )
    ap.add_argument("--ongoing-rounds", type=int, default=600)
    ap.add_argument("--ongoing-min-leaf", type=int, default=100)
    ap.add_argument(
        "--stack-onset", action="store_true", help="second-stage onset model on out-of-fold blended probabilities + spatial context"
    )
    ap.add_argument("--stack-thr", type=float, default=0.45)
    ap.add_argument("--stack-folds", type=int, default=5)
    ap.add_argument("--ongoing-obs-gap0", action="store_true", help="ongoing horizon labels from the raw observation, gap cells weighted 0")
    ap.add_argument(
        "--conds", nargs="*", default=["onset", "ongoing"], help="conditions to fit and predict (pred.csv then holds only these windows)"
    )
    ap.add_argument(
        "--onset-obs-gap0",
        action="store_true",
        help="onset labels at T+30 from the raw observation, detector-gap cells weighted 0 (the U-Net field misses fresh queues)",
    )
    a = ap.parse_args()
    t0 = time.time()
    out = OUT / f"final_{a.name}"
    out.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(windows_path(a.truth))
    pqs = {pn: PanelQ(pn, truth=a.truth) for pn in QUEUE_PANELS}
    statics = {pn: static_feats(pqs[pn]) for pn in QUEUE_PANELS}
    fit_days = {pn: np.arange(len(pqs[pn].dates)) for pn in QUEUE_PANELS}
    clim = Climatology(pqs, fit_days, onset_windows=w[w.condition == "onset"])
    clim_fit = clim
    if os.environ.get("QUEUE_LOMC") in ("1", "onset", "ongoing"):
        months = sorted(w.month.unique())
        clim_fit = {
            m: Climatology(pqs, {pn: np.where(pqs[pn].months != m)[0] for pn in QUEUE_PANELS}, onset_windows=w[w.condition == "onset"])
            for m in months
        }
        print(f"leave-own-month-out climatologies built for {len(clim_fit)} months ({time.time() - t0:.0f}s)", flush=True)
    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbose": -1,
        "seed": 42,
        "num_threads": 10,
    }
    models = {}
    for cond in a.conds:
        if a.reuse and (out / f"{cond}.txt").exists():
            models[cond] = [
                lgb.Booster(model_file=str(f)) for f in sorted(out.glob(f"{cond}*.txt")) if "raw" not in f.name and "noclim" not in f.name
            ]
            print(f"{cond}: reused {len(models[cond])} saved model(s)")
            if cond == "onset" and a.onset_hedge:
                models["onset_raw"] = [lgb.Booster(model_file=str(f)) for f in sorted(out.glob("onset_raw*.txt"))]
            wnc = a.noclim_w_onset if cond == "onset" else a.noclim_w_ongoing
            ppm = os.environ.get("QUEUE_PERPANEL", "") if cond == "ongoing" else ""
            tag = {"1": "_pp", "2": "_pf"}.get(ppm, "")
            pp = sorted(out.glob(f"ongoing_noclim{tag}*_*.txt")) if tag else []
            if wnc > 0 and pp:  # per-panel / per-family no-clim models (QUEUE_PERPANEL=1/2)
                models["ongoing_noclim_pp"] = {}
                for fp in pp:
                    models["ongoing_noclim_pp"].setdefault(int(fp.stem.split(tag)[1].split("_")[0]), []).append(
                        lgb.Booster(model_file=str(fp))
                    )
                nf = models[cond][0].feature_name()
                ip = nf.index("panel")
                models["ongoing_noclim_idx"] = [i for i in noclim_idx(nf) if ppm == "2" or i != ip]
                models["ongoing_noclim"] = []
                print(f"ongoing: reused per-panel models for {len(models['ongoing_noclim_pp'])} panels", flush=True)
            elif wnc > 0 and list(out.glob(f"{cond}_noclim*.txt")):
                models[f"{cond}_noclim"] = [lgb.Booster(model_file=str(f)) for f in sorted(out.glob(f"{cond}_noclim*.txt"))]
                models[f"{cond}_noclim_idx"] = noclim_idx(models[cond][0].feature_name())
            continue
        X, y, keys, names = build_rows(
            w,
            pqs,
            clim_fit if os.environ.get("QUEUE_LOMC") in ("1", cond) else clim,
            statics,
            cond,
            OUT
            / f"feat_{cond}_fit_all{'' if a.feats == 'v1' else '_' + a.feats}{'' if a.truth == 'raw' else '_' + a.truth}{'_cap' + os.environ['INMONTH_CAP'] if os.environ.get('INMONTH_CAP') else ''}{'_fill' + os.environ['QUEUE_FILL'] if os.environ.get('QUEUE_FILL') not in (None, '', '1') else ('_fill' if os.environ.get('QUEUE_FILL') == '1' else '')}{'_nodrop' if os.environ.get('QUEUE_NODROP') == '1' else ''}{'_ft' + os.environ['QUEUE_FILL_TRAIN'] if os.environ.get('QUEUE_FILL_TRAIN') else ''}{'_lomc' + os.environ['QUEUE_LOMC'] if os.environ.get('QUEUE_LOMC') in ('1', cond) else ''}{FC_TAG()}{DIR_TAG()}.npz",
            a.feats,
            "regime",
        )
        pr = (
            {**params, "seed": 42}
            if cond == "onset"
            else {**params, "num_leaves": a.ongoing_leaves, "min_data_in_leaf": a.ongoing_min_leaf}
        )
        rounds = a.rounds if cond == "onset" else a.ongoing_rounds
        wt = None
        if (cond == "onset" and a.onset_obs_gap0) or (
            cond == "ongoing" and a.ongoing_obs_gap0
        ):  # labels: raw observation where observed, gaps weight 0
            from tfb.labels.observed import cell_info

            obs30, rq30 = cell_info(w, keys)
            y = np.where(obs30, rq30, y).astype(np.int8)
            wt = obs30.astype(float)
            print(f"{cond} obs-gap0: gap share {1 - obs30.mean():.3f}, positives {int(y.sum())}", flush=True)
        if (
            os.environ.get("QUEUE_WINWEIGHT") and cond == "ongoing"
        ):  # per-window weights (c:alpha): rows x (window positives + c)^-alpha, so each
            c_ww, a_ww = (
                float(v) for v in os.environ["QUEUE_WINWEIGHT"].split(":")
            )  # each window counts alike, as in the official mean of window IoUs
            _, inv_ww = np.unique(keys[:, 0], return_inverse=True)
            ww = (np.bincount(inv_ww, y.astype(float))[inv_ww] + c_ww) ** (-a_ww)
            ww = ww / ww.mean()
            wt = ww if wt is None else wt * ww
            print(f"{cond}: per-window weights c {c_ww} alpha {a_ww} (min {ww.min():.3f} max {ww.max():.3f})", flush=True)
        n_tr, wt_tr = len(y), wt
        if os.environ.get("QUEUE_ADAPT"):  # append selector-replica windows of an evaluation split (scenario adaptation;
            asplit, aw = os.environ["QUEUE_ADAPT"].split(":")
            aw = float(aw)  # use the resulting predictions for LATER splits only)
            zr = np.load(OUT / f"feat_{cond}_replica_{asplit}_{a.feats}_mix_fillcausal_nodrop.npz", allow_pickle=True)
            assert [str(n) for n in zr["names"]] == list(names), "replica feature names differ"
            X = np.concatenate([X, zr["X"]])
            y = np.concatenate([y, zr["y"]]).astype(np.int8)
            wt = np.concatenate([np.ones(n_tr) if wt is None else wt, np.full(len(zr["y"]), aw)])
            print(f"{cond}: + {len(zr['y'])} replica rows of {asplit} at weight {aw}", flush=True)
        models[cond] = [fit_model(X, y, names, {**pr, "seed": 42 + i}, rounds, weight=wt) for i in range(a.seeds)]
        for i, m in enumerate(models[cond]):
            m.save_model(str(out / f"{cond}{'' if i == 0 else i}.txt"))
        wnc = a.noclim_w_onset if cond == "onset" else a.noclim_w_ongoing
        pp_on = (cond == "ongoing" and os.environ.get("QUEUE_PERPANEL") in ("1", "2")) or (
            cond == "onset" and os.environ.get("QUEUE_PERPANEL_ONSET") == "1"
        )
        if wnc > 0 and pp_on:  # per panel (1) or per corridor family (2, ongoing only)
            fam = cond == "ongoing" and os.environ.get("QUEUE_PERPANEL") == "2"
            leaves = (
                int(os.environ.get("QUEUE_PP_LEAVES", "255")) if cond == "ongoing" else int(os.environ.get("QUEUE_PP_LEAVES_ONSET", "31"))
            )
            ip = names.index("panel")
            keep = [i for i in noclim_idx(names) if fam or i != ip]
            nn = [names[i] for i in keep]
            models[f"{cond}_noclim_pp"] = {}
            grp = (X[:, ip] // 2) if fam else X[:, ip]
            for pidv in np.unique(grp):
                mm = grp == pidv
                models[f"{cond}_noclim_pp"][int(pidv)] = [
                    fit_model(
                        X[mm][:, keep],
                        y[mm],
                        nn,
                        {**pr, "num_leaves": leaves, "seed": 42 + i},
                        rounds,
                        weight=None if wt is None else wt[mm],
                    )
                    for i in range(a.seeds)
                ]
                for i, m in enumerate(models[f"{cond}_noclim_pp"][int(pidv)]):
                    m.save_model(str(out / f"{cond}_noclim_{'pf' if fam else 'pp'}{int(pidv)}_{i}.txt"))
                print(f"{cond} per-panel model {int(pidv)}: {int(mm.sum())} rows ({time.time() - t0:.0f}s)", flush=True)
            models[f"{cond}_noclim_idx"] = keep
            models[f"{cond}_noclim"] = []
        elif wnc > 0:
            keep = noclim_idx(names)
            nn = [names[i] for i in keep]
            models[f"{cond}_noclim"] = [fit_model(X[:, keep], y, nn, {**pr, "seed": 42 + i}, rounds, weight=wt) for i in range(a.seeds)]
            for i, m in enumerate(models[f"{cond}_noclim"]):
                m.save_model(str(out / f"{cond}_noclim{'' if i == 0 else i}.txt"))
            models[f"{cond}_noclim_idx"] = keep
            print(f"{cond}: no-climatology models trained ({time.time() - t0:.0f}s)", flush=True)
        if cond == "onset" and a.onset_hedge and a.truth != "raw":
            from tfb.common import CACHE

            rawq = {pn: np.load(CACHE / pn / "train_queued.npy", mmap_mode="r") for pn in QUEUE_PANELS}
            y_raw = np.zeros(len(keys), np.int8)
            for widx in np.unique(keys[:, 0]):
                sel = keys[:, 0] == widx
                r = w.loc[widx]
                y_raw[sel] = rawq[r["panel"]][int(r["d"]), int(r["T"]) + HOR][keys[sel, 1]]
            if a.onset_obs_gap0:
                y_raw = np.where(obs30, rq30, y_raw).astype(np.int8)
            models["onset_raw"] = [
                fit_model(X[:n_tr], y_raw, names, {**pr, "seed": 42 + i}, rounds, weight=wt_tr) for i in range(a.seeds)
            ]  # train rows only
            for i, m in enumerate(models["onset_raw"]):
                m.save_model(str(out / f"onset_raw{'' if i == 0 else i}.txt"))
            print(
                f"onset hedge: raw-label models trained, positives raw {int(y_raw.sum())} vs {int(y.sum())} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        print(f"{cond}: rows {X.shape} positives {int(y.sum())} trained ({time.time() - t0:.0f}s)", flush=True)
    if a.stack_onset:
        from tfb.common import CACHE
        from tfb.queue.onset_stack import STACK_KEEP, context

        sp = out / "onset_stack.txt"
        Xo, yo, ko, no = build_rows(
            w,
            pqs,
            clim_fit,
            statics,
            "onset",
            OUT
            / f"feat_onset_fit_all{'' if a.feats == 'v1' else '_' + a.feats}{'' if a.truth == 'raw' else '_' + a.truth}{'_cap' + os.environ['INMONTH_CAP'] if os.environ.get('INMONTH_CAP') else ''}{'_fill' + os.environ['QUEUE_FILL'] if os.environ.get('QUEUE_FILL') not in (None, '', '1') else ('_fill' if os.environ.get('QUEUE_FILL') == '1' else '')}{'_nodrop' if os.environ.get('QUEUE_NODROP') == '1' else ''}{'_ft' + os.environ['QUEUE_FILL_TRAIN'] if os.environ.get('QUEUE_FILL_TRAIN') else ''}{'_lomc' + os.environ['QUEUE_LOMC'] if os.environ.get('QUEUE_LOMC') in ('1', cond) else ''}{FC_TAG()}{DIR_TAG()}.npz",
            a.feats,
            "regime",
        )
        models["onset_stack_keep"] = [i for i, n in enumerate(no) if n in STACK_KEEP]
        if a.reuse and sp.exists():
            models["onset_stack"] = lgb.Booster(model_file=str(sp))
            print("onset stacker reused", flush=True)
        else:
            pr_on = {**params, "seed": 42}
            rawq = {pn: np.load(CACHE / pn / "train_queued.npy", mmap_mode="r") for pn in QUEUE_PANELS}
            yo_raw = np.zeros(len(ko), np.int8)
            for widx in np.unique(ko[:, 0]):
                sel = ko[:, 0] == widx
                r = w.loc[widx]
                yo_raw[sel] = rawq[r["panel"]][int(r["d"]), int(r["T"]) + HOR][ko[sel, 1]]
            wins = np.unique(ko[:, 0])
            rng = np.random.default_rng(0)
            fold_of = dict(zip(wins, rng.integers(0, a.stack_folds, len(wins))))
            fold = np.array([fold_of[i] for i in ko[:, 0]])
            p_oof = np.zeros(len(yo))
            keep = noclim_idx(no)
            nn = [no[i] for i in keep]
            for f in range(a.stack_folds):
                tr = fold != f
                te = ~tr
                pf = np.mean(
                    [fit_model(Xo[tr], yo[tr], no, {**pr_on, "seed": 42 + i}, a.rounds).predict(Xo[te]) for i in range(a.seeds)], 0
                )
                if a.onset_hedge and a.truth != "raw":
                    pf = np.sqrt(
                        pf
                        * np.mean(
                            [
                                fit_model(Xo[tr], yo_raw[tr], no, {**pr_on, "seed": 42 + i}, a.rounds).predict(Xo[te])
                                for i in range(a.seeds)
                            ],
                            0,
                        )
                    )
                if a.noclim_w_onset > 0:
                    pnc = np.mean(
                        [
                            fit_model(Xo[tr][:, keep], yo[tr], nn, {**pr_on, "seed": 42 + i}, a.rounds).predict(Xo[te][:, keep])
                            for i in range(a.seeds)
                        ],
                        0,
                    )
                    pf = (1 - a.noclim_w_onset) * pf + a.noclim_w_onset * pnc
                p_oof[te] = pf
                print(f"onset stack fold {f} done ({time.time() - t0:.0f}s)", flush=True)
            S = np.concatenate([context(p_oof, ko, Xo, no), Xo[:, models["onset_stack_keep"]]], 1)
            sn = ["p", "p_up1", "p_dn1", "p_up2", "p_dn2", "p_max", "p_sum", "n35", "rank", "p_rel", "p_nb_max", "n20", "p_second", "L"] + [
                no[i] for i in models["onset_stack_keep"]
            ]
            stk = {**params, "num_leaves": 31, "min_data_in_leaf": 200, "seed": 42}
            models["onset_stack"] = lgb.train(
                stk,
                lgb.Dataset(S, yo, feature_name=sn, categorical_feature=["panel"] if "panel" in sn else [], free_raw_data=False),
                num_boost_round=400,
            )
            models["onset_stack"].save_model(str(sp))
            print(f"onset stacker trained on {len(yo)} out-of-fold rows ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    probs = {}
    clim_priv = None
    if os.environ.get("QUEUE_CLIM_PRIV"):  # private split: pooled train + March climatology (all before April)
        import pickle

        with open(os.environ["QUEUE_CLIM_PRIV"], "rb") as f:
            clim_priv = pickle.load(f)
        print("private split uses", os.environ["QUEUE_CLIM_PRIV"], flush=True)
    for split in a.splits:
        for pn in QUEUE_PANELS:
            pq = PanelQ(pn, split)
            wi = released_windows(pn, split)
            roll = os.environ.get("QUEUE_INM_ROLL", "")
            if roll == "1" or (
                roll == "eval" and split == "private"
            ):  # rolling recurrence: the previous split's last 16 days (all before d)
                prev = pqs[pn] if split == "validation" else PanelQ(pn, "validation")
                pq.prev_speed = np.asarray(prev.speed[-16:], np.float32)
            for r in wi.itertuples():
                d = pq.dates.index(r.date)
                T = int(r.T)
                L = pq.p.n_links
                cond = r.condition.replace("queue_", "")
                if cond not in a.conds:
                    continue
                restore = pq.patch(d, T)
                Xf, _ = link_feats(pq, d, T, clim_priv if (clim_priv is not None and split == "private") else clim, *statics[pn])
                if a.feats in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"):
                    Xo, _ = origin_feats(pq, d, T, None, None)
                    Xt, _ = trend_feats(pq, d, T)
                    Xf = np.concatenate([Xf, Xo, Xt], 1)
                    if a.feats in ("v3", "v15"):
                        Xd, _ = dayctx_feats(pq, d, T, None, None)
                        Xf = np.concatenate([Xf, Xd], 1)
                    if a.feats == "v4":
                        Xb, _ = balance_feats(pq, d, T)
                        Xf = np.concatenate([Xf, Xb], 1)
                    if a.feats == "v5":
                        Xa, _ = abs_speed_feats(pq, d, T)
                        Xf = np.concatenate([Xf, Xa], 1)
                    if a.feats in ("v6", "v8", "v9", "v10", "v11", "v12", "v13", "v15", "v16", "v17"):
                        Xi, _ = inmonth_feats(pq, d, T, None, None, cond)
                        Xf = np.concatenate([Xf, Xi], 1)
                    if a.feats in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"):
                        Xc, _ = corridor_feats(pq, d, T)
                        Xf = np.concatenate([Xf, Xc], 1)
                    if a.feats in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17"):
                        Xc2, _ = corridor_feats2(pq, d, T)
                        Xf = np.concatenate([Xf, Xc2], 1)
                    if a.feats == "v11":
                        Xsh, _ = shock_feats(pq, d, T)
                        Xf = np.concatenate([Xf, Xsh], 1)
                    if a.feats == "v12":
                        Xb3, _ = corridor_feats3(pq, d, T)
                        Xf = np.concatenate([Xf, Xb3], 1)
                    if a.feats == "v13":
                        Xeg, _ = edge_feats(pq, d, T)
                        Xf = np.concatenate([Xf, Xeg], 1)
                    if a.feats in ("v16", "v17"):
                        Xmd, _ = inmonth_day_feats(pq, d, T)
                        Xf = np.concatenate([Xf, Xmd], 1)
                    if a.feats == "v17":
                        Xmq, _ = inmonth_demand_feats(pq, d, T, cond)
                        Xf = np.concatenate([Xf, Xmq], 1)
                    if a.feats == "v8":
                        Xo8, _ = inmonth_onset_feats(pq, d, T, None, None, cond)
                        Xf = np.concatenate([Xf, Xo8], 1)
                    if a.feats == "v7":
                        Xr, _ = remcap_feats(pq, d, T)
                        Xf = np.concatenate([Xf, Xr], 1)
                restore()
                fc = pq.forecast(d, T)
                if fc is not None:
                    Xf = np.concatenate([Xf, fc_link_feats(fc, _dir(pq))], 1)
                    if FC_FEATS() == 2:
                        fq, fk = pq.forecast_qk(d, T)
                        Xf = np.concatenate([Xf, fc_link_feats2(fc, fq, fk, _dir(pq))], 1)
                    if FC_FEATS() == 3:
                        fp = pq.forecast_p(d, T)
                        Xf = np.concatenate([Xf, fc_link_feats3(fp, _dir(pq))], 1)
                pid = np.full((L, 1), PANEL_ID[pn], float)
                pred = np.zeros((HOR, L), dtype=bool)
                if cond == "onset":
                    Xin = np.concatenate([Xf, pid], 1).astype(np.float32)
                    prob = np.mean([m.predict(Xin) for m in models["onset"]], 0)
                    if "onset_raw" in models:
                        prob = np.sqrt(prob * np.mean([m.predict(Xin) for m in models["onset_raw"]], 0))
                    if "onset_noclim" in models:
                        keep = models["onset_noclim_idx"]
                        mlist = models["onset_noclim_pp"][int(PANEL_ID[pn])] if "onset_noclim_pp" in models else models["onset_noclim"]
                        pnc = np.mean([m.predict(Xin[:, keep]) for m in mlist], 0)
                        prob = (1 - a.noclim_w_onset) * prob + a.noclim_w_onset * pnc
                    if "onset_stack" in models:
                        S1 = np.concatenate([context(prob, np.zeros((L, 3), int), None, None), Xin[:, models["onset_stack_keep"]]], 1)
                        prob = models["onset_stack"].predict(S1)
                    s = decode(prob, a.onset_mode, thr=(a.stack_thr if "onset_stack" in models else a.onset_thr))
                    pred[-1, s] = True
                    probs[r.window_id] = prob.astype(np.float32)
                else:
                    hc = horizon_clim(pq, T, clim)
                    ih = inmonth_horizon(pq, d, T) if INMH_ON() else None
                    Xh = np.concatenate(
                        [
                            np.concatenate(
                                [Xf]
                                + ([fc_h_feats(fc, h, _dir(pq))] if fc is not None else [])
                                + ([fc_h_feats2(fc, fq, fk, h)] if fc is not None and FC_FEATS() == 2 else [])
                                + ([fc_h_feats3(fp, h, _dir(pq))] if fc is not None and FC_FEATS() == 3 else [])
                                + ([ih[h - 1][:, None]] if ih is not None else [])
                                + [np.column_stack([np.full(L, float(h)), hc[h - 1], pid[:, 0]])],
                                1,
                            )
                            for h in range(1, HOR + 1)
                        ]
                    ).astype(np.float32)
                    prob = np.mean([m.predict(Xh) for m in models["ongoing"]], 0)
                    if "ongoing_noclim" in models:
                        keep = models["ongoing_noclim_idx"]
                        gk = int(PANEL_ID[pn]) // (2 if os.environ.get("QUEUE_PERPANEL") == "2" else 1)
                        mlist = models["ongoing_noclim_pp"][gk] if "ongoing_noclim_pp" in models else models["ongoing_noclim"]
                        pnc = np.mean([m.predict(Xh[:, keep]) for m in mlist], 0)
                        prob = (1 - a.noclim_w_ongoing) * prob + a.noclim_w_ongoing * pnc
                    s = decode(prob, a.ongoing_mode, thr=a.ongoing_thr, max_k=2000)
                    pred = s.reshape(HOR, L)
                    probs[r.window_id] = prob.astype(np.float32).reshape(HOR, L)
                for h in range(HOR):
                    ts = iso(r.date, T + 1 + h)
                    rows.append(
                        pd.DataFrame({"window_id": r.window_id, "timestamp": ts, "link_id": pq.p.links, "queue_pred": pred[h].astype(int)})
                    )
            print(f"{split} {pn}: {len(wi)} windows done ({time.time() - t0:.0f}s)", flush=True)
    pred_df = pd.concat(rows, ignore_index=True)
    pred_df.to_csv(out / "pred.csv", index=False)
    if a.dump_probs:
        np.savez(out / "probs.npz", **probs)
    summary = pred_df.groupby(pred_df.window_id.str.replace(r"_\d+$", "", regex=True)).queue_pred.sum()
    (out / "meta.json").write_text(
        json.dumps({"args": vars(a), "rows": len(pred_df), "positives": int(pred_df.queue_pred.sum())}, indent=1)
    )
    print("positives per panel-split:", summary.to_dict())
    print("total rows", len(pred_df), "positives", int(pred_df.queue_pred.sum()), f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
