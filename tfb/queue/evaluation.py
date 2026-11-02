"""Queue evaluation harness on the selector-matched train windows (forward protocol)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tfb.common import CACHE, PROJECT, QUEUE_PANELS, load_panel
from tfb.queue.data import HIST, HOR, released_windows
from tfb.scoring import queue_aggregate

WINDOWS = PROJECT / "results_top3" / "queue" / "windows_train.csv"


def occ_density_ratio(panel: str) -> np.ndarray:
    """Per-link occupancy / (flow / speed) from the train split's observed cells (cached)."""
    f = CACHE / panel / "occ_density_ratio.npy"
    if f.exists():
        return np.load(f)
    p = load_panel(panel, "train")
    sp = np.asarray(p["speed"], np.float32)
    fl = np.asarray(p["flow"], np.float32)
    oc = np.asarray(p["occ"], np.float32)
    k = fl / np.maximum(sp, 1.0)
    with np.errstate(all="ignore"):
        ok = np.isfinite(oc) & np.isfinite(k) & (k > 0) & (oc > 0)
        ratio = np.where(ok, oc / np.where(k > 0, k, np.nan), np.nan)
        c = np.nanmedian(ratio.reshape(-1, ratio.shape[2]), axis=0)
    c = np.where(np.isfinite(c), c, np.nanmedian(c)).astype(np.float32)
    np.save(f, c)
    return c


def windows_path(truth: str = "raw") -> Path:
    """Path of the train windows table of a truth version (results_top3/queue/windows_train[_<truth>].csv)."""
    return PROJECT / "results_top3" / "queue" / f"windows_train{'' if truth == 'raw' else '_' + truth}.csv"


TEST_MONTHS = ("2030-12", "2031-01", "2031-02")


class PanelQ:
    """Everything a queue predictor may look at for one panel (train split)."""

    def __init__(self, panel: str, split: str = "train", truth: str = "raw"):
        self.panel = panel
        self.split = split
        self.truth_name = truth
        self.p = load_panel(panel, split)
        import os

        # QUEUE_INFER=1: inference with saved models -- the train panel then only feeds the climatology (labels) and the static
        # features, so its fill fields, causal fills and forecasts (training inputs) are not loaded
        infer_train = split == "train" and os.environ.get("QUEUE_INFER") == "1"
        if split == "train":
            suf = "" if truth == "raw" else f"_{truth}"
            self.truth = np.load(CACHE / panel / f"train_queued{suf}.npy", mmap_mode="r")
            self.speed_f = (
                None
                if infer_train
                else np.load(CACHE / panel / ("train_speed_filled.npy" if truth == "raw" else f"train_speed_{truth}.npy"), mmap_mode="r")
            )
            self.speed = self.p["speed"]
            self.flow = self.p["flow"]
            self.occ = self.p["occ"]
        else:
            self.truth = None
            self.speed_f = None
            self.speed = self.p["speed_m"]
            self.flow = self.p["flow_m"]
            self.occ = self.p["occ_m"]
        if os.environ.get("QUEUE_FILL") == "1" and not infer_train:  # fill detector gaps of the history with the U-Net field
            sp = np.asarray(self.speed, np.float32)
            if split == "train":
                fill = np.load(CACHE / panel / f"train_speed_{os.environ.get('QUEUE_FILL_TRAIN', 'dn')}.npy", mmap_mode="r")
            else:
                fill = np.load(
                    PROJECT / "results_top3" / "state" / os.environ.get("QUEUE_FILL_STATE", "enslongC") / f"{panel}_{split}.npz"
                )["speed"]
            self.speed = np.where(np.isfinite(sp), sp, np.asarray(fill, np.float32))
        if (
            os.environ.get("QUEUE_FILL") in ("2", "3", "causal") and not infer_train
        ):  # speed and flow gaps filled (3: occupancy derived as well)
            sp = np.asarray(self.speed, np.float32)
            fl = np.asarray(self.flow, np.float32)
            if split == "train":
                src = os.environ.get("QUEUE_FILL_TRAIN", "dn2")
                fs = np.load(CACHE / panel / f"train_speed_{src}.npy", mmap_mode="r")
                fq = np.load(CACHE / panel / f"train_flow_{src}.npy", mmap_mode="r")
            else:
                z = np.load(PROJECT / "results_top3" / "state" / os.environ.get("QUEUE_FILL_STATE", "enslongC") / f"{panel}_{split}.npz")
                fs, fq = z["speed"], z["flow"]
            self.speed = np.where(np.isfinite(sp), sp, np.asarray(fs, np.float32))
            self.flow = np.where(np.isfinite(fl), fl, np.asarray(fq, np.float32))
            if os.environ.get("QUEUE_FILL") in ("3", "causal"):  # occupancy ~ c_link * flow / speed
                oc = np.asarray(self.occ, np.float32)
                k = self.flow / np.maximum(self.speed, 1.0)
                if os.environ.get("QUEUE_FILL") == "causal":
                    c = occ_density_ratio(panel)  # from the train split only (legal for every split)
                else:
                    with np.errstate(all="ignore"):
                        ok = np.isfinite(oc) & np.isfinite(k) & (k > 0) & (oc > 0)
                        ratio = np.where(ok, oc / np.where(k > 0, k, np.nan), np.nan)
                        c = np.nanmedian(ratio.reshape(-1, ratio.shape[2]), axis=0)
                    c = np.where(np.isfinite(c), c, np.nanmedian(c))
                self.occ = np.where(np.isfinite(oc), oc, (k * c[None, None, :]).astype(np.float32))
                self._c_occ = c
        self.obs = np.isfinite(self.speed) & (self.speed <= self.p.param("v_cut")[None, None, :])
        self._causal = None
        if os.environ.get("QUEUE_FILL") == "causal" and not infer_train:  # per-window fills with everything after T hidden
            z = np.load(CACHE / panel / f"{split}_causalfill.npz")
            self._causal = {(int(dd), int(tt)): i for i, (dd, tt) in enumerate(zip(z["d"], z["T"]))}
            self._cv = z["v"]
            self._cq = z["q"]
            self._occ_hist = self.p["occ"] if split == "train" else self.p["occ_m"]
            self._occ_T = self.p["occ_m"]
        self._prefix = None
        if os.environ.get("QUEUE_CAUSAL_PREFIX") == "1" and split != "train":  # evaluation: the whole same-day prefix causally filled
            zp = np.load(CACHE / panel / f"{split}_causalprefix.npz")
            self._prefix = {(int(dd), int(tt)): i for i, (dd, tt) in enumerate(zip(zp["d"], zp["T"]))}
            self._pv = zp["v"]
            self._pq = zp["q"]
        self._fc = None
        if os.environ.get("QUEUE_FC") and not infer_train:  # per-window U-Net forecasts for T+5..T+30 (legal: causal inputs only)
            sfx = (
                os.environ["QUEUE_FC"] if split == "train" else os.environ.get("QUEUE_FC_EVAL", os.environ["QUEUE_FC"])
            )  # eval: one fold model per run
            zf = np.load(CACHE / panel / f"{split}_causalfill{sfx}.npz")
            self._fc = {(int(dd), int(tt)): i for i, (dd, tt) in enumerate(zip(zf["d"], zf["T"]))}
            self._fcv = zf["hv"]
            self._fcq = zf["hq"]
            self._fcp = zf["hp"] if "hp" in zf.files else None  # queue-head forecasters only
        self.elig = self.p["elig"]
        self.vcut = self.p.param("v_cut")
        self.cap = self.p.param("capacity")
        self.dates = self.p.dates
        self.months = np.array([d[:7] for d in self.dates])

    def patch(self, d: int, T: int):
        """Causal mode: swap rows [T-HIST, T] of day d for the window's causal fill; returns the restore callable."""
        if self._causal is None:
            return lambda: None
        if self._prefix is not None:
            j = self._prefix[(int(d), int(T))]
            rows = slice(0, T + 1)
            saved = (self.speed[d, rows].copy(), self.flow[d, rows].copy(), self.occ[d, rows].copy(), self.obs[d, rows].copy())
            v = self._pv[j][: T + 1]
            q = self._pq[j][: T + 1]
            oc = np.asarray(self._occ_T[d, : T + 1], np.float32)
            oc = np.where(np.isfinite(oc), oc, (q / np.maximum(v, 1.0)) * self._c_occ[None, :])
            self.speed[d, rows] = v
            self.flow[d, rows] = q
            self.occ[d, rows] = oc
            self.obs[d, rows] = v <= self.vcut[None, :]

            def restore_prefix():
                """Write the saved rows 0..T of day d back into speed, flow, occ and obs."""
                self.speed[d, rows], self.flow[d, rows], self.occ[d, rows], self.obs[d, rows] = saved

            return restore_prefix
        i = self._causal[(int(d), int(T))]
        rows = slice(T - HIST, T + 1)
        saved = (self.speed[d, rows].copy(), self.flow[d, rows].copy(), self.occ[d, rows].copy(), self.obs[d, rows].copy())
        v = self._cv[i]
        q = self._cq[i]
        oc = np.concatenate([np.asarray(self._occ_hist[d, T - HIST : T], np.float32), np.asarray(self._occ_T[d, T : T + 1], np.float32)])
        oc = np.where(np.isfinite(oc), oc, (q / np.maximum(v, 1.0)) * self._c_occ[None, :])
        self.speed[d, rows] = v
        self.flow[d, rows] = q
        self.occ[d, rows] = oc
        self.obs[d, rows] = v <= self.vcut[None, :]

        def restore():
            """Write the saved rows T-HIST..T of day d back into speed, flow, occ and obs."""
            self.speed[d, rows], self.flow[d, rows], self.occ[d, rows], self.obs[d, rows] = saved

        return restore

    def forecast(self, d: int, T: int):
        """(HOR, L) forecast speed ratio to v_cut, or None when forecasts are off."""
        if self._fc is None:
            return None
        return self._fcv[self._fc[(int(d), int(T))]] / self.vcut[None, :]

    def forecast_p(self, d: int, T: int):
        """(HOR, L) forecast queue probability (feature set 3; needs a queue-head forecast file)."""
        return self._fcp[self._fc[(int(d), int(T))]]

    def forecast_qk(self, d: int, T: int):
        """(HOR, L) forecast flow / capacity and density / k_crit (feature set 2)."""
        i = self._fc[(int(d), int(T))]
        hv, hq = self._fcv[i], self._fcq[i]
        return hq / self.cap[None, :], (hq / np.maximum(hv, 1.0)) / self.p.param("k_crit")[None, :]

    def hist_flags(self, d: int, T: int) -> np.ndarray:
        """(HIST, L) observed queue flags of day d over the history T-HIST..T-1."""
        return self.obs[d, T - HIST : T]

    def last_eligible_flag(self, d: int, T: int) -> np.ndarray:
        """Queue flag of each link's last eligible observation in the history (False if none)."""
        h = self.obs[d, T - HIST : T]
        e = self.elig[d, T - HIST : T] & np.isfinite(self.speed[d, T - HIST : T])
        out = np.zeros(h.shape[1], dtype=bool)
        for l in range(h.shape[1]):
            idx = np.where(e[:, l])[0]
            if len(idx):
                out[l] = h[idx[-1], l]
        return out

    def truth_window(self, d: int, T: int) -> np.ndarray:
        """(HOR, L) truth queue flags of day d at T+1..T+HOR (train split)."""
        return np.asarray(self.truth[d, T + 1 : T + 1 + HOR])


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """IoU of two boolean arrays; 1.0 when both are empty."""
    u = (pred | truth).sum()
    return 1.0 if u == 0 else float((pred & truth).sum() / u)


def evaluate(predictors: dict, windows: pd.DataFrame, panels: dict[str, PanelQ]) -> pd.DataFrame:
    """One row (model, panel, condition, d, T, iou) per window and predictor; a predictor maps (pq, d, T, condition) -> (HOR, L)."""
    rows = []
    for r in windows.itertuples():
        pq = panels[r.panel]
        tr = pq.truth_window(r.d, r.T)
        for name, fn in predictors.items():
            pred = fn(pq, r.d, r.T, r.condition)
            rows.append({"model": name, "panel": r.panel, "condition": r.condition, "d": r.d, "T": r.T, "iou": iou(pred.astype(bool), tr)})
    return pd.DataFrame(rows)


def report(res: pd.DataFrame, title: str) -> None:
    """Print per model: S_queue, the per-condition means and each condition's lowest per-panel mean IoU."""
    print(f"\n=== {title} ===")
    for name, g in res.groupby("model", sort=False):
        agg = queue_aggregate(g)
        pc = g.groupby(["panel", "condition"]).iou.mean().unstack()
        worst = pc.min().to_dict()
        print(
            f"{name:32s} S_queue {agg['S_queue']:.4f} | onset {agg['per_condition'].get('onset', float('nan')):.4f} "
            f"ongoing {agg['per_condition'].get('ongoing', float('nan')):.4f} | worst panel onset {worst.get('onset', float('nan')):.3f} ongoing {worst.get('ongoing', float('nan')):.3f}"
        )


# ---------------- baselines
def zeros(pq, d, T, cond):
    """Baseline: no queue, (HOR, L) all False."""
    return np.zeros((HOR, pq.p.n_links), dtype=bool)


def persist_last(pq, d, T, cond):
    """Baseline: the observed queue flags at T-1 repeated over the horizon."""
    return np.broadcast_to(pq.obs[d, T - 1][None, :], (HOR, pq.p.n_links)).copy()


def persist_last_elig(pq, d, T, cond):
    """Baseline: each link's last eligible history flag (last_eligible_flag) repeated over the horizon."""
    return np.broadcast_to(pq.last_eligible_flag(d, T)[None, :], (HOR, pq.p.n_links)).copy()


def union_hist(pq, d, T, cond):
    """Baseline: every link with a queued observation in the history, queued over the whole horizon."""
    return np.broadcast_to(pq.hist_flags(d, T).any(0)[None, :], (HOR, pq.p.n_links)).copy()


class Climatology:
    """freq[slot, link] of the truth queue flag over the fit days; onset prediction = top-k at T+30."""

    def __init__(self, panels: dict[str, PanelQ], fit_months: tuple[str, ...]):
        self.freq = {}
        for name, pq in panels.items():
            days = np.where(np.isin(pq.months, fit_months))[0]
            self.freq[name] = np.asarray(pq.truth[days]).mean(0)  # (288, L)

    def onset_topk(self, k: int):
        """Onset predictor: at T+30 only, the k links with the highest freq at slot T+HOR (freq > 0)."""

        def fn(pq, d, T, cond):
            """(HOR, L) flags: the last step only, on the top-k links by freq at slot T+HOR with freq > 0."""
            out = np.zeros((HOR, pq.p.n_links), dtype=bool)
            f = self.freq[pq.panel][T + HOR]
            top = np.argsort(-f)[:k]
            out[-1, top[f[top] > 0]] = True
            return out

        return fn


def main() -> None:
    w = pd.read_csv(WINDOWS)
    panels = {pn: PanelQ(pn) for pn in QUEUE_PANELS}
    fit_months = tuple(m for m in sorted(w.month.unique()) if m not in TEST_MONTHS)
    clim = Climatology(panels, fit_months)
    test = w[w.month.isin(TEST_MONTHS)]
    print("test windows", len(test), test.groupby("condition").size().to_dict())
    preds = {"zeros": zeros, "persist_last": persist_last, "persist_last_elig": persist_last_elig, "union_hist": union_hist}
    for k in (1, 2, 3, 4, 5, 6, 8):
        preds[f"clim_top{k}+persist_elig"] = lambda pq, d, T, c, k=k: (
            clim.onset_topk(k)(pq, d, T, c) if c == "onset" else persist_last_elig(pq, d, T, c)
        )
    res = evaluate(preds, test, panels)
    report(res, "forward test (Dec-Feb), fit on Jun-Nov")
    res.to_csv(PROJECT / "results_top3" / "queue" / "baselines_forward.csv", index=False)
    # released 80 windows (June): fit on everything except June
    rel = []
    for pn in QUEUE_PANELS:
        r = released_windows(pn, "train")
        pq = panels[pn]
        rel.append(
            pd.DataFrame(
                {
                    "panel": pn,
                    "d": [pq.dates.index(x) for x in r.date],
                    "T": r["T"].astype(int),
                    "condition": r.condition.str.replace("queue_", ""),
                    "month": r.date.str.slice(0, 7),
                }
            )
        )
    rel = pd.concat(rel, ignore_index=True)
    clim2 = Climatology(panels, tuple(m for m in sorted(w.month.unique()) if m != "2030-06"))
    preds2 = {"persist_last_elig": persist_last_elig, "union_hist": union_hist}
    for k in (2, 3, 4, 6):
        preds2[f"clim_top{k}+persist_elig"] = lambda pq, d, T, c, k=k: (
            clim2.onset_topk(k)(pq, d, T, c) if c == "onset" else persist_last_elig(pq, d, T, c)
        )
    report(evaluate(preds2, rel, panels), "released 80 June windows, climatology fit on Jul-Feb")


if __name__ == "__main__":
    main()
