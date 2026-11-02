"""How much do the evaluation months' 90-minute blackouts cost the Task-1 reconstruction?

In validation/private every scored Task-2 window is followed by a blackout: all mainline links are blank for the horizon
and the sixty minutes after it (steps T+1..T+18), while the quality fields stay and the Task-1 template still lists the
hash-masked cells inside (1.8 % of all targets on D12_I5_S in March). The train masked layer has no blackouts, so holdout31
never scored such cells. This script places blackouts on the holdout31 days at greedy-spaced released train windows
(>= 360 min apart, like the evaluation selector), un-masks the published raw history rows T-12..T-1 (not Task-1 targets in the
release), and compares the member's errors with and without the blackouts on the same cells.

Run: python -m tfb.state.blackout_eval --ckpts results_top3/state/unet_h31_long46/best.pt [--device cpu]
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tfb.common import FLOW_NORM, FLOW_W, PANELS, PROJECT, SPEED_NORM, SPEED_W
from tfb.scoring import aggregate_panels
from tfb.state.data import holdout_dates, load_arrays
from tfb.state.predict import ensemble_predict, load_nets

HIST, BLACK, SPACING = 12, 18, 72


def blackout_windows(pn: str, dates: list[str], days: np.ndarray, seed: int | None = None) -> list[tuple[int, int]]:
    """Greedy >= 360-min spaced (day index, T) per day from the released train window candidates.

    seed None: greedy in time order (the evaluation selector's way); otherwise the candidates are visited in a random order
    (another valid placement, for more corrector training data)."""
    w = pd.read_csv(PROJECT / "results_top3" / "queue" / "windows_train_mix.csv")
    w = w[w.panel == pn]
    rng = np.random.default_rng(seed) if seed is not None else None
    out = []
    for d in days:
        ts = sorted(w[w.date == dates[d]]["T"].astype(int).unique())
        if rng is not None:
            chosen = []
            for T in rng.permutation(ts):
                if T >= HIST and T + BLACK < 288 and all(abs(int(T) - c) >= SPACING for c in chosen):
                    chosen.append(int(T))
            out.extend((int(d), T) for T in sorted(chosen))
            continue
        last = -(10**9)
        for T in ts:
            if T - last >= SPACING and T >= HIST and T + BLACK < 288:
                out.append((int(d), int(T)))
                last = T
    return out


def score(a, pv, pq, days, cells) -> tuple[float, float, float]:
    """Pooled per-regime S_state on a boolean cell mask (len(days), 288, L), averaged over regimes present."""
    p = a.panel
    lanes = p.param("lanes")[None, None, :]
    tv, tq = p["speed"][days], p["flow"][days]
    reg = p.regimes[days]
    S, ev_all, eq_all = [], [], []
    for r in sorted(set(reg)):
        sel = reg == r
        m = cells[sel]
        if not m.any():
            continue
        ev = (pv[sel] - tv[sel])[m]
        eq = ((pq[sel] - tq[sel]) / np.broadcast_to(lanes, tq[sel].shape))[m]
        rv = float(np.sqrt(np.mean(ev.astype(np.float64) ** 2)))
        rq = float(np.sqrt(np.mean(eq.astype(np.float64) ** 2)))
        S.append(SPEED_W * max(0.0, 1 - rv / SPEED_NORM) + FLOW_W * max(0.0, 1 - rq / FLOW_NORM))
        ev_all.append(ev)
        eq_all.append(eq)
    ev = np.concatenate(ev_all)
    eq = np.concatenate(eq_all)
    return float(np.mean(S)), float(np.sqrt(np.mean(ev.astype(np.float64) ** 2))), float(np.sqrt(np.mean(eq.astype(np.float64) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--dump", default="", help="save per-panel predictions (npz dir)")
    a_ = ap.parse_args()
    torch.set_num_threads(6)
    dev = torch.device(a_.device)
    nets = load_nets(a_.ckpts, dev)
    h31 = holdout_dates()
    SA, SB, SBe = {}, {}, {}
    tot_in = tot_all = 0
    for pn in a_.panels:
        a = load_arrays(pn, "train", with_trans=False)
        days = np.array([a.dates.index(d) for d in h31 if d in a.dates])
        wins = blackout_windows(pn, a.dates, days)
        pos = {d: i for i, d in enumerate(days)}
        v, q, occ, tgt = a.v.copy(), a.q.copy(), a.occ.copy(), a.target.copy()
        black = np.zeros((len(days), 288), bool)
        hist = np.zeros((len(days), 288), bool)
        for d, T in wins:
            black[pos[d], T + 1 : T + 1 + BLACK] = True
            hist[pos[d], T - HIST : T] = True
            v[d, T + 1 : T + 1 + BLACK] = np.nan
            q[d, T + 1 : T + 1 + BLACK] = np.nan
            occ[d, T + 1 : T + 1 + BLACK] = np.nan
            tgt[d, T - HIST : T] = False  # published raw history T-12..T-1: observed, not a target
        b = dataclasses.replace(a, v=v, q=q, occ=occ, target=tgt)
        pvA, pqA = ensemble_predict(nets, a, days, dev)
        pvB, pqB = ensemble_predict(nets, b, days, dev)
        cells = a.panel["target"][days] & a.panel["elig"][days] & ~hist[:, :, None]
        inb = cells & black[:, :, None]
        sA, rvA, _rqA = score(a, pvA, pqA, days, cells)
        sB, rvB, _rqB = score(a, pvB, pqB, days, cells)
        _, rvAi, rqAi = score(a, pvA, pqA, days, inb)
        _, rvBi, rqBi = score(a, pvB, pqB, days, inb)
        # reweight the blackout cells to the evaluation share (1.83 %) by subsampling them
        share = inb.sum() / cells.sum()
        rng = np.random.default_rng(0)
        keep = rng.random(inb.shape) < min(1.0, 0.0183 / max(share, 1e-9))
        cells_e = cells & (~inb | keep)
        sAe, _, _ = score(a, pvA, pqA, days, cells_e)
        sBe, _, _ = score(a, pvB, pqB, days, cells_e)
        SA[pn], SB[pn], SBe[pn] = sA, sB, sBe - sAe
        tot_in += int(inb.sum())
        tot_all += int(cells.sum())
        print(
            f"{pn}: {len(wins)} blackouts, blackout targets {inb.sum()} ({share:.2%}) | inside rmse_v {rvAi:.2f} -> {rvBi:.2f}, "
            f"rmse_q {rqAi:.1f} -> {rqBi:.1f} | all rmse_v {rvA:.3f} -> {rvB:.3f} | S_state {sA:.5f} -> {sB:.5f} "
            f"(at 1.83 % share: {sBe - sAe:+.5f})",
            flush=True,
        )
        if a_.dump:
            Path(a_.dump).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(a_.dump) / f"{pn}.npz", days=days, pvA=pvA, pqA=pqA, pvB=pvB, pqB=pqB, black=black, hist=hist)
    print(
        f"aggregate S_state {aggregate_panels(SA):.5f} -> {aggregate_panels(SB):.5f} (blackout share {tot_in / tot_all:.2%}); "
        f"at the evaluation share: {aggregate_panels(SBe):+.5f} S_state = {0.35 * aggregate_panels(SBe):+.5f} total"
    )


if __name__ == "__main__":
    main()
