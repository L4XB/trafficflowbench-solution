"""Vectorised local scorers on the array cache.

State (Task 1) reproduces the official evaluator exactly. Physics (Task 3) uses
an anchored-flux approximation: with the organizer's flux
projected onto the released observations the right-hand side equals the truth's
own increment, so E_LWR = sum|dN_hat - dN_true| / sum|dN_true| over the usable
transitions. It ranks reconstructions; the official absolute level differs
(a perfect answer scores ~0.96 there, 1.0 here). Queue (Task 2) is the official
space-time IoU with the official aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfb.common import FAMILY, FLOW_NORM, FLOW_W, REGIMES, SPEED_NORM, SPEED_W, STEPS, Panel

EMPTY_FLOOR_VPH, EMPTY_SHARE = 50.0, 0.20


def state_score_panel(
    p: Panel, pred_v: np.ndarray, pred_q: np.ndarray, days: np.ndarray, truth_v: np.ndarray | None = None, truth_q: np.ndarray | None = None
) -> dict:
    """S_state for one panel over ``days`` (indices into the split).

    ``pred_v/pred_q`` are (len(days), 288, L) predictions; truth defaults to the
    cached unmasked train layer. Only target & eligible cells are scored.
    """
    if truth_v is None:
        truth_v, truth_q = p["speed"][days], p["flow"][days]
    lanes = p.param("lanes")[None, None, :]
    target = p["target"][days] & p["elig"][days]
    reg = p.regimes[days]
    out = {}
    for r in REGIMES:
        sel = reg == r
        if not sel.any():
            out[r] = None
            continue
        m = target[sel]
        ev = (pred_v[sel] - truth_v[sel])[m]
        eq = ((pred_q[sel] - truth_q[sel]) / np.broadcast_to(lanes, truth_q[sel].shape))[m]
        rv = float(np.sqrt(np.mean(ev.astype(np.float64) ** 2)))
        rq = float(np.sqrt(np.mean(eq.astype(np.float64) ** 2)))
        out[r] = {
            "rmse_v": rv,
            "rmse_q": rq,
            "S": SPEED_W * max(0.0, 1 - rv / SPEED_NORM) + FLOW_W * max(0.0, 1 - rq / FLOW_NORM),
            "n": int(m.sum()),
        }
    present = [v["S"] for v in out.values() if v is not None]
    out["S_state"] = float(np.mean(present)) if present else 0.0
    return out


def _ramp_validity(p: Panel, days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(on_valid, off_valid) per (day, t, link): structural 1 for unattached links."""
    relig = p["ramp_elig"][days]  # (D, T, R)
    rflow = p["ramp_flow"][days]
    valid = relig & np.isfinite(rflow)
    rtype = np.asarray(p.meta["ramps"]["ramp_type"])
    rlink = np.asarray(p.meta["ramps"]["ramp_link"])
    L = p.n_links
    on_valid = np.zeros(valid.shape[:2] + (L,), dtype=bool)
    off_valid = np.zeros_like(on_valid)
    for i, (typ, l) in enumerate(zip(rtype, rlink)):
        if l < 0:
            continue
        if typ.upper() in ("OR", "ON"):
            on_valid[:, :, l] |= valid[:, :, i]
        elif typ.upper() in ("FR", "OFF"):
            off_valid[:, :, l] |= valid[:, :, i]
    on_att = np.asarray(p.meta["params"].get("on_attached", [False] * L), dtype=bool)
    off_att = np.asarray(p.meta["params"].get("off_attached", [False] * L), dtype=bool)
    on_valid |= ~on_att[None, None, :]
    off_valid |= ~off_att[None, None, :]
    return on_valid, off_valid


def physics_score_panel(
    p: Panel, pred_v: np.ndarray, pred_q: np.ndarray, days: np.ndarray, truth_v: np.ndarray | None = None, truth_q: np.ndarray | None = None
) -> dict:
    """Anchored-flux S_physics for one panel over ``days`` (sorted split indices)."""
    if truth_v is None:
        truth_v, truth_q = p["speed"][days], p["flow"][days]
    dates = np.asarray(p.dates)[days]
    reg = p.regimes[days]
    elig = p["elig"][days]
    target = p["target"][days] & elig
    lanes = p.param("lanes")
    length = p.param("length_km")
    vf = p.param("free_speed")
    cap = p.param("capacity")
    kj = p.param("k_jam")
    kc = cap / np.maximum(vf, 1.0)
    kj = np.maximum(kj, kc * 1.05)
    on_valid, off_valid = _ramp_validity(p, days)
    out = {}
    for r in REGIMES:
        sel = np.where(reg == r)[0]
        if len(sel) == 0:
            out[r] = None
            continue
        tv = np.where(target[sel], pred_v[sel], truth_v[sel]).astype(np.float64)
        tq = np.where(target[sel], pred_q[sel], truth_q[sel]).astype(np.float64)
        e = elig[sel]
        k = tq / np.maximum(tv, 1.0)
        # --- FD, per lane, over all eligible frame cells
        ql = tq / lanes
        kl = k / lanes
        capl = cap / lanes
        kcl = kc / lanes
        kjl = kj / lanes
        qfd = np.where(kl <= kcl, vf * kl, (capl / np.maximum(kjl - kcl, 1e-9)) * np.maximum(kjl - kl, 0))
        s_fd = max(0.0, 1 - np.abs(ql - qfd)[e].sum() / (np.abs(ql)[e].sum() + 1e-9))
        sub = target[sel]
        if sub.any() and (tq[sub] < EMPTY_FLOOR_VPH).mean() > EMPTY_SHARE:
            s_fd = 0.0
        # --- LWR: transitions between consecutive eligible cells 5 minutes apart
        N = k * length
        Nt = (truth_q[sel].astype(np.float64) / np.maximum(truth_v[sel], 1.0)) * length
        # concatenate days in order; a cross-day transition is valid only for consecutive dates
        d_sel = dates[sel]
        consecutive = np.array(
            [(pd.Timestamp(str(d_sel[i + 1])) - pd.Timestamp(str(d_sel[i]))).days == 1 for i in range(len(d_sel) - 1)] + [False]
        )
        D = len(sel)
        Nf = N.reshape(D * STEPS, -1)
        Ntf = Nt.reshape(D * STEPS, -1)
        ef = e.reshape(D * STEPS, -1)
        onf = on_valid[sel].reshape(D * STEPS, -1)
        offf = off_valid[sel].reshape(D * STEPS, -1)
        ok_step = np.ones(D * STEPS - 1, dtype=bool)
        # step from t=287 of day i to t=0 of day i+1
        for i in range(D - 1):
            ok_step[i * STEPS + STEPS - 1] = consecutive[i]
        use = ef[:-1] & ef[1:] & ok_step[:, None] & onf[:-1] & offf[:-1]
        dN = (Nf[1:] - Nf[:-1])[use]
        dNt = (Ntf[1:] - Ntf[:-1])[use]
        num = np.abs(dN - dNt).sum()
        den = np.abs(dNt).sum() + 1e-9
        e_lwr = num / den
        s_lwr = max(0.0, 1 - min(1.0, e_lwr))
        out[r] = {
            "S_FD": float(s_fd),
            "S_LWR": float(s_lwr),
            "E_LWR": float(e_lwr),
            "S_physics": float((s_fd + 2 * s_lwr) / 3),
            "n_trans": int(use.sum()),
        }
    present = [v["S_physics"] for v in out.values() if v is not None]
    out["S_physics"] = float(np.mean(present)) if present else 0.0
    return out


def aggregate_panels(scores: dict[str, float]) -> float:
    """Panel -> family mean -> mean over the families present."""
    fam: dict[str, list[float]] = {}
    for p, s in scores.items():
        fam.setdefault(FAMILY[p], []).append(s)
    return float(np.mean([np.mean(v) for v in fam.values()]))


def iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    """Space-time IoU over the eligible cells only; 1.0 when neither has a queued eligible cell."""
    pr = pred.astype(bool) & elig
    tr = truth.astype(bool) & elig
    union = (pr | tr).sum()
    return 1.0 if union == 0 else float((pr & tr).sum() / union)


def queue_aggregate(rows: pd.DataFrame) -> dict:
    """rows: panel, condition, iou (one per window) -> official aggregation."""
    pc = rows.groupby(["panel", "condition"]).iou.mean().reset_index()
    pm = pc.groupby("panel").iou.mean()
    fam = pm.groupby(pm.index.map(FAMILY)).mean()
    return {
        "S_queue": float(fam.mean()),
        "per_panel": pm.to_dict(),
        "per_condition": pc.groupby("condition").iou.mean().to_dict(),
        "per_panel_condition": {(r.panel, r.condition): r.iou for r in pc.itertuples()},
    }
