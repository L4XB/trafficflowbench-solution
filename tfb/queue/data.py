"""Queue truth rebuilt from the unmasked train layer, and the Task-2 window selector replica."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfb.common import RELEASE, STEPS, Panel, load_panel, t_index

HIST, HOR, SPACING = 12, 6, 72  # steps


def fill_time(grid: np.ndarray) -> np.ndarray:
    """(T, L) NaN-filled by linear interpolation along T per link (edges extended)."""
    out = grid.astype(np.float64).copy()
    T = grid.shape[0]
    x = np.arange(T, dtype=float)
    for l in range(grid.shape[1]):
        c = out[:, l]
        ok = np.isfinite(c)
        if ok.all() or not ok.any():
            continue
        out[:, l] = np.interp(x, x[ok], c[ok])
    return out


def filled_speed(p: Panel) -> np.ndarray:
    """Truth speed with detector gaps interpolated in time (approximates the noise-free field)."""
    sp = p["speed"]
    out = np.empty(sp.shape, dtype=np.float32)
    vf = p.param("free_speed")
    for d in range(sp.shape[0]):
        f = fill_time(sp[d])
        dead = ~np.isfinite(f).any(axis=0)
        if dead.any():
            f[:, dead] = vf[dead][None, :]
        out[d] = f
    return out


def queued_truth(p: Panel, speed_filled: np.ndarray | None = None) -> np.ndarray:
    """(days, 288, links) truth queue flag: filled speed <= v_cut (speed_filled if given, else filled_speed(p))."""
    if speed_filled is None:
        speed_filled = filled_speed(p)
    return speed_filled <= p.param("v_cut")[None, None, :]


def queued_observed(p: Panel) -> np.ndarray:
    """Queue flag on the raw observation (NaN -> False)."""
    sp = p["speed"]
    return np.isfinite(sp) & (sp <= p.param("v_cut")[None, None, :])


def window_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """IoU of two boolean arrays; 1.0 when both are empty."""
    u = (pred | truth).sum()
    return 1.0 if u == 0 else float((pred & truth).sum() / u)


def candidates(
    p: Panel,
    truth: np.ndarray,
    obs: np.ndarray,
    elig: np.ndarray,
    condition: str,
    hist_rule: str = "zero",
    require_last_queued: bool = True,
) -> list[tuple[int, int, dict]]:
    """All valid origins (d, T) for a condition, before spacing. Returns (d, T, info)."""
    D = truth.shape[0]
    out = []
    for d in range(D):
        for T in range(HIST, STEPS - HOR):
            h_obs = obs[d, T - HIST : T]  # T-60..T-5 observed queue flags
            h_el = elig[d, T - HIST : T]
            cov = h_el.mean()
            if cov < 0.7:
                continue
            hor = truth[d, T + 1 : T + 1 + HOR]  # T+5..T+30 on the (approximately) noise-free field
            if not hor.any():
                continue
            per_link = h_obs.sum(0)
            established = (per_link >= 2).any()
            if condition == "onset":
                if hist_rule == "zero" and h_obs.any():
                    continue
                if hist_rule == "contract" and established:
                    continue
                if hor[:-1].any():
                    continue
                out.append((d, T, {"n30": int(hor[-1].sum())}))
            else:
                if not established:
                    continue
                last = h_obs[-1]
                if require_last_queued and not last.any():
                    continue
                pers = np.broadcast_to(last[None, :], hor.shape)
                piou = window_iou(pers, hor)
                if piou > 0.9:
                    continue
                out.append((d, T, {"piou": piou, "npos": int(hor.sum())}))
    return out


def greedy(cands: list[tuple[int, int, dict]], limit: int | None = None) -> list[tuple[int, int, dict]]:
    """Greedy 360-minute spacing over the whole split timeline."""
    sel, last = [], -(10**9)
    for d, T, info in cands:  # cands are in (d, T) order
        a = d * STEPS + T
        if a - last >= SPACING:
            sel.append((d, T, info))
            last = a
            if limit and len(sel) >= limit:
                break
    return sel


def released_windows(panel: str, split: str) -> pd.DataFrame:
    """The released window index (task2/<panel>/<split>/window_index.csv) plus the origin slot T from forecast_origin."""
    w = pd.read_csv(RELEASE / "task2" / panel / split / "window_index.csv")
    w["T"] = t_index(w.forecast_origin.to_numpy())
    return w


if __name__ == "__main__":
    from tfb.common import QUEUE_PANELS

    for panel in QUEUE_PANELS:
        p = load_panel(panel, "train")
        sf = filled_speed(p)
        truth = queued_truth(p, sf)
        obs = queued_observed(p)
        elig = p["elig"]
        w = released_windows(panel, "train")
        for cond in ("onset", "ongoing"):
            rel = w[w.condition == f"queue_{cond}"]
            rel_set = [(p.dates.index(r.date), int(r.T)) for r in rel.itertuples()]
            for rule in (("zero", True), ("contract", True), ("contract", False)):
                c = candidates(p, truth, obs, elig, cond, hist_rule=rule[0], require_last_queued=rule[1])
                g = greedy(c, limit=5)
                gset = [(d, T) for d, T, _ in g]
                valid = sum((d, T) in {(x[0], x[1]) for x in c} for d, T in rel_set)
                print(
                    f"{panel:10s} {cond:8s} rule={rule} cands={len(c):5d} greedy5={gset} released={rel_set} exact={gset == rel_set} released_valid={valid}/5"
                )
                if cond == "onset":
                    break
