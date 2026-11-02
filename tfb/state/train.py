"""Train the Task-1 inpainting U-Net.

Run (development protocol, holdout31 + February 2031 excluded from training):
    python -m tfb.state.train --name unet_s --minutes 20
Run (final model on all 273 train days, fixed number of epochs):
    python -m tfb.state.train --name unet_final --exclude none --epochs 60

The loss mirrors the objective: 0.35 * (0.54 RMSE_v/25 + 0.46 RMSE_ql/600) on the
synthetically blanked eligible cells plus w_phys * E_LWR, where E_LWR is the
anchored-flux density-increment error of tfb/scoring.py evaluated on the batch.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch

from tfb.common import PANELS, PROJECT
from tfb.scoring import aggregate_panels, physics_score_panel, state_score_panel
from tfb.state.data import (
    FLOW_SCALE,
    N_FEATURES,
    SPEED_MAX,
    SPEED_SCALE,
    PanelArrays,
    TrainSampler,
    batch_inputs,
    february_dates,
    full_panel_batch,
    holdout_dates,
    load_arrays,
)
from tfb.state.model import build_model

OUT = PROJECT / "results_top3" / "state"
W_STATE_OBJ, W_PHYS_OBJ = 0.35, 0.15


def parse() -> argparse.Namespace:
    """Command-line options of a U-Net training run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude",
        default="both",
        choices=["both", "holdout31", "feb", "none", "fold0", "fold1", "fold2", "fold3"],
        help="days excluded from training",
    )
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps_per_epoch", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--minutes", type=float, default=0.0, help="wall-clock budget for training (0 = epochs only)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--w_phys", type=float, default=0.10)
    ap.add_argument("--state_loss", default="mse", choices=["mse", "rmse", "l1"])
    ap.add_argument("--phys_loss", default="huber", choices=["l1", "huber", "none"])
    ap.add_argument("--huber_delta", type=float, default=0.05, help="Huber threshold on |dN_hat-dN_true| / mean|dN_true|")
    ap.add_argument("--res_scale_v", type=float, default=10.0, help="km/h per unit of the speed head output")
    ap.add_argument("--res_scale_q", type=float, default=200.0, help="vph/lane per unit of the flow head output")
    ap.add_argument("--eval_raw", action="store_true", help="also evaluate the raw (non-EMA) weights")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--phys_warmup", type=int, default=0, help="steps over which w_phys ramps linearly from 0")
    ap.add_argument("--p_band", type=float, default=0.2)
    ap.add_argument("--band_windows", default="", help="windows csv (panel,date,T): place bands right after window origins (blackout mode)")
    ap.add_argument("--p_band_win", type=float, default=0.8, help="share of bands placed at a window origin when the day has one")
    ap.add_argument("--band_loss", action="store_true", help="every eligible band cell is a loss cell")
    ap.add_argument("--rate_jitter", type=float, default=0.1)
    ap.add_argument("--panel_power", type=float, default=0.5, help="panel sampling weight ~ L**power")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--resume", default="")
    ap.add_argument("--forecast", action="store_true", help="train a 30-minute forecaster (everything after a random origin hidden)")
    return ap.parse_args()


def predict_full(net: torch.nn.Module, a: PanelArrays, days: np.ndarray, device, chunk: int = 8):
    """Predict (len(days), 288, L) speed and flow with the real masks in ``a``."""
    net.eval()
    pv, pq = [], []
    with torch.no_grad():
        for i in range(0, len(days), chunk):
            s = full_panel_batch(a, days[i : i + chunk], device)
            x, v_lin, ql_lin = batch_inputs(s)
            out = net(x)
            rv, rq = getattr(net, "res_scale", (SPEED_SCALE, FLOW_SCALE))
            v = (v_lin + out[:, 0] * rv).clamp(0, SPEED_MAX)
            ql = (ql_lin + out[:, 1] * rq).clamp(min=0)
            q = ql * s["lanes"].unsqueeze(1)
            L = s["n_links"]
            pv.append(v[:, :, :L].float().cpu().numpy())
            pq.append(q[:, :, :L].float().cpu().numpy())
    net.train()
    return np.concatenate(pv).astype(np.float32), np.concatenate(pq).astype(np.float32)


def evaluate(net, panels: list[PanelArrays], day_sets: dict[str, dict[str, np.ndarray]], device) -> dict:
    """Scores per holdout set: per-panel state/physics dicts and aggregates."""
    res = {}
    for hname, dmap in day_sets.items():
        S, P, per = {}, {}, {}
        for a in panels:
            days = dmap[a.name]
            if len(days) == 0:
                continue
            pv, pq = predict_full(net, a, days, device)
            s = state_score_panel(a.panel, pv, pq, days)
            ph = physics_score_panel(a.panel, pv, pq, days)
            S[a.name] = s["S_state"]
            P[a.name] = ph["S_physics"]
            per[a.name] = {"state": s, "physics": ph}
        ss, sp = aggregate_panels(S), aggregate_panels(P)
        res[hname] = {"S_state": ss, "S_physics": sp, "objective": W_STATE_OBJ * ss + W_PHYS_OBJ * sp, "per_panel": per}
    return res


def evaluate_forecast(
    net, panels: list[PanelArrays], day_sets: dict[str, dict[str, np.ndarray]], device, origins=(96, 120, 144, 168, 192, 216)
) -> dict:
    """Forecast quality on the holdout days: everything after T hidden, speed RMSE and queue IoU over T+5..T+30."""
    import dataclasses

    res = {}
    for hname, dmap in day_sets.items():
        se, n, inter, union = 0.0, 0, 0, 0
        for a in panels:
            days = dmap[a.name]
            if len(days) == 0:
                continue
            vc = a.panel.param("v_cut").astype(np.float32)
            dd = np.repeat(days, len(origins))
            TT = np.tile(np.array(origins), len(days))
            v = a.v[dd].copy()
            q = a.q[dd].copy()
            occ = a.occ[dd].copy()
            pct = a.pct[dd].copy()
            el = a.elig[dd].copy()
            ron = a.ramp_on[dd].copy()
            roff = a.ramp_off[dd].copy()
            rok = a.ramp_on_ok[dd].copy()
            fok = a.ramp_off_ok[dd].copy()
            truth = a.v[dd].copy()
            for i, T in enumerate(TT):
                v[i, T + 1 :] = np.nan
                q[i, T + 1 :] = np.nan
                occ[i, T + 1 :] = np.nan
                pct[i, T + 1 :] = 0
                el[i, T + 1 :] = False
                ron[i, T + 1 :] = 0
                roff[i, T + 1 :] = 0
                rok[i, T + 1 :] = False
                fok[i, T + 1 :] = False
            b = dataclasses.replace(
                a,
                dates=[a.dates[x] for x in dd],
                v=v,
                q=q,
                occ=occ,
                pct=pct,
                elig=el,
                target=np.zeros(v.shape, bool),
                ramp_on=ron,
                ramp_off=roff,
                ramp_on_ok=rok,
                ramp_off_ok=fok,
                weekday=a.weekday[dd],
                trans_ok=None,
            )
            pv, _ = predict_full(net, b, np.arange(len(dd)), device)
            for i, T in enumerate(TT):
                tr = truth[i, T + 1 : T + 7]
                pr = pv[i, T + 1 : T + 7]
                fin = np.isfinite(tr)
                se += float(((pr - tr)[fin] ** 2).sum())
                n += int(fin.sum())
                qt = fin & (tr <= vc[None, :])
                qp = pr <= vc[None, :]
                inter += int((qt & qp).sum())
                union += int((qt | qp).sum())
        rmse = (se / max(n, 1)) ** 0.5
        iou = inter / max(union, 1)
        res[hname] = {
            "S_state": iou,
            "S_physics": rmse,
            "objective": iou - rmse / 100.0,
            "fc_rmse_v": rmse,
            "fc_queue_iou": iou,
            "per_panel": {},
        }
    return res


def loss_fn(out: torch.Tensor, v_lin: torch.Tensor, ql_lin: torch.Tensor, s: dict, args) -> tuple:
    """W_STATE_OBJ x state loss on the loss cells (--state_loss) + w_phys x physics loss on E_LWR (--phys_loss); returns (total, logs)."""
    m = s["loss_mask"]
    v_pred = (v_lin + out[:, 0] * args.res_scale_v).clamp(0, SPEED_MAX)
    ql_pred = (ql_lin + out[:, 1] * args.res_scale_q).clamp(min=0)
    B, _T, W = v_pred.shape
    lanes = s["lanes"].view(B, 1, W)
    length = s["length"].view(B, 1, W)
    v = torch.nan_to_num(s["v"])
    q = torch.nan_to_num(s["q"])
    ql = q / lanes
    n = m.sum().clamp(min=1)
    mse_v = ((v_pred - v) ** 2 * m).sum() / n
    mse_q = ((ql_pred - ql) ** 2 * m).sum() / n
    rmse_v = torch.sqrt(mse_v + 1e-6)
    rmse_q = torch.sqrt(mse_q + 1e-6)
    if args.state_loss == "rmse":
        state = 0.54 * rmse_v / 25.0 + 0.46 * rmse_q / 600.0
    elif args.state_loss == "l1":  # median forecasts (sharper at queue fronts); same score-unit weighting as the RMSE option
        mae_v = ((v_pred - v).abs() * m).sum() / n
        mae_q = ((ql_pred - ql).abs() * m).sum() / n
        state = 0.54 * mae_v / 25.0 + 0.46 * mae_q / 600.0
    else:  # MSE in score units; d(RMSE/norm)/d(MSE/norm^2) = 1/(2 RMSE/norm) ~ 4 at the baseline error level
        state = 4.0 * (0.54 * mse_v / 25.0**2 + 0.46 * mse_q / 600.0**2)
    # physics: density increment error over counted transitions (numerator/denominator of E_LWR)
    k_true = q / v.clamp(min=1.0)
    k_hat = (ql_pred * lanes) / v_pred.clamp(min=1.0)
    e = torch.where(m, (k_hat - k_true) * length, torch.zeros_like(k_hat))
    n_true = k_true * length
    use = s["trans"]
    de = (e[:, 1:] - e[:, :-1]).abs() * use
    den = ((n_true[:, 1:] - n_true[:, :-1]).abs() * use).sum().clamp(min=1e-3)
    e_lwr = de.sum() / den
    if args.phys_loss == "l1":
        phys = e_lwr
    elif args.phys_loss == "huber":
        scale = den / use.sum().clamp(min=1)  # mean |dN_true| per counted transition
        z = de / scale
        d = args.huber_delta
        hub = torch.where(z < d, 0.5 * z**2 / d, z - 0.5 * d) * use
        phys = hub.sum() * scale / den
    else:
        phys = torch.zeros((), device=out.device)
    w_phys = args.w_phys * min(1.0, (getattr(args, "_step", 0) + 1) / args.phys_warmup) if args.phys_warmup > 0 else args.w_phys
    total = W_STATE_OBJ * state + w_phys * phys
    return total, {"rmse_v": rmse_v.item(), "rmse_q": rmse_q.item(), "e_lwr": e_lwr.item(), "state": state.item()}


def main() -> None:
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(os.environ.get("TRAIN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
    run = OUT / args.name
    run.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["features"] = N_FEATURES
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(run / "log.jsonl", "a")  # noqa: SIM115  (appended to once per epoch for the whole run)

    t0 = time.time()
    panels = [load_arrays(p, "train") for p in args.panels]
    h31 = holdout_dates()
    feb = february_dates()
    folds = {f"fold{k}": {d for i, d in enumerate(panels[0].dates) if i % 4 == k} for k in range(4)}  # out-of-fold fills
    excl = {"both": set(h31) | set(feb), "holdout31": set(h31), "feb": set(feb), "none": set(), **folds}[args.exclude]
    train_days = {a.name: np.array([i for i, d in enumerate(a.dates) if d not in excl]) for a in panels}
    day_sets = {
        "holdout31": {a.name: np.array([a.dates.index(d) for d in h31 if d in a.dates]) for a in panels},
        "feb2031": {a.name: np.array([a.dates.index(d) for d in feb if d in a.dates]) for a in panels},
    }
    print(
        f"loaded {len(panels)} panels in {time.time() - t0:.0f}s; train days/panel {len(train_days[panels[0].name])}; "
        f"excluded {len(excl)} dates",
        flush=True,
    )

    band_windows = None
    if args.band_windows:
        import pandas as pd

        wdf = pd.read_csv(args.band_windows)
        band_windows = {}
        for a in panels:
            di = {dt: i for i, dt in enumerate(a.dates)}
            ww = wdf[(wdf.panel == a.name) & (wdf["T"] >= 12) & (wdf["T"] <= 288 - 19)]
            m: dict = {}
            for dt, T in zip(ww.date, ww["T"].astype(int)):
                if dt in di:
                    m.setdefault(di[dt], []).append(T)
            band_windows[a.name] = {k: sorted(set(v)) for k, v in m.items()}
        print("band windows:", {k: sum(len(v) for v in m.values()) for k, m in band_windows.items()}, flush=True)
    sampler = TrainSampler(
        panels,
        train_days,
        args.crop,
        args.batch,
        device,
        forecast=args.forecast,
        p_band=args.p_band,
        rate_jitter=args.rate_jitter,
        seed=args.seed,
        band_windows=band_windows,
        p_band_win=args.p_band_win,
        band_loss=args.band_loss,
    )
    w = np.array([max(a.n_links, args.crop) ** args.panel_power for a in panels])
    sampler.pw = w / w.sum()
    mcfg = {"widths": args.widths, "blocks": args.blocks, "bottleneck_dilations": args.dilations}
    mcfg["res_scale"] = [args.res_scale_v, args.res_scale_q]
    net = build_model(mcfg, N_FEATURES).to(device)
    net.res_scale = (args.res_scale_v, args.res_scale_q)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"model params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))
    total_steps = args.epochs * args.steps_per_epoch
    if args.minutes > 0:  # estimate the schedule length from the time budget after the first epoch
        total_steps = None
    step = 0
    best = -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        step = ck.get("step", 0)
        if args.forecast or args.band_windows:
            step = 0  # fine-tuning: fresh schedule
        print("resumed from", args.resume, "step", step)

    def lr_at(st: int, tot: int) -> float:
        """Learning rate at step st of tot: linear warmup over --warmup steps, then cosine decay from --lr to 1e-5."""
        if st < args.warmup:
            return args.lr * (st + 1) / args.warmup
        pr = min(1.0, (st - args.warmup) / max(1, tot - args.warmup))
        return 1e-5 + (args.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * pr))

    t_train0 = time.time()
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        acc = {"loss": 0.0, "rmse_v": 0.0, "rmse_q": 0.0, "e_lwr": 0.0}
        for _ in range(args.steps_per_epoch):
            tot = total_steps if total_steps else max(step + 1, args.steps_per_epoch * 2)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tot)
            s = sampler.sample()
            x, v_lin, ql_lin = batch_inputs(s)
            args._step = step
            out = net(x)
            loss, parts = loss_fn(out, v_lin, ql_lin, s, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(args.ema).add_(pn.detach(), alpha=1 - args.ema)
                for be, bn in zip(ema.buffers(), net.buffers()):
                    be.copy_(bn)
            step += 1
            acc["loss"] += loss.item()
            for k in ("rmse_v", "rmse_q", "e_lwr"):
                acc[k] += parts[k]
            if not math.isfinite(loss.item()) or loss.item() == 0.0:
                print("ABORT: loss", loss.item(), "at step", step, flush=True)
                return
        for k in acc:
            acc[k] /= args.steps_per_epoch
        train_min = (time.time() - t_train0) / 60
        if args.minutes > 0 and total_steps is None:
            per_epoch = (time.time() - te) / 60
            total_steps = int(args.minutes / per_epoch) * args.steps_per_epoch
            print(f"time budget {args.minutes} min -> {total_steps // args.steps_per_epoch} epochs", flush=True)
        rec = {"epoch": epoch, "step": step, "train": acc, "lr": opt.param_groups[0]["lr"], "min": round(train_min, 1)}
        if epoch % args.eval_every == 0 or step >= total_steps:
            ev = (evaluate_forecast if args.forecast else evaluate)(ema, panels, day_sets, device)
            rec["eval"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_panel"} for k, v in ev.items()}
            if args.eval_raw:
                evr = evaluate(net, panels, day_sets, device)
                rec["eval_raw"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_panel"} for k, v in evr.items()}
                print(
                    f"      raw: h31 S_state {evr['holdout31']['S_state']:.5f} S_phys {evr['holdout31']['S_physics']:.5f} "
                    f"obj {evr['holdout31']['objective']:.5f} | feb obj {evr['feb2031']['objective']:.5f}",
                    flush=True,
                )
            obj = ev["holdout31"]["objective"]
            if obj > best:
                best = obj
                torch.save(
                    {
                        "model": net.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "cfg": cfg,
                        "mcfg": mcfg,
                        "eval": ev,
                    },
                    run / "best.pt",
                )
                rec["saved"] = True
            h, f = ev["holdout31"], ev["feb2031"]
            print(
                f"ep {epoch:3d} step {step:6d} {train_min:5.1f}min loss {acc['loss']:.4f} rmse_v {acc['rmse_v']:.3f} "
                f"rmse_q {acc['rmse_q']:.1f} elwr {acc['e_lwr']:.3f} | h31 S_state {h['S_state']:.5f} S_phys {h['S_physics']:.5f} "
                f"obj {h['objective']:.5f} | feb S_state {f['S_state']:.5f} S_phys {f['S_physics']:.5f} obj {f['objective']:.5f}"
                f"{' *' if rec.get('saved') else ''}  ({time.time() - te:.0f}s)",
                flush=True,
            )
        else:
            print(
                f"ep {epoch:3d} step {step:6d} {train_min:5.1f}min loss {acc['loss']:.4f} rmse_v {acc['rmse_v']:.3f} "
                f"rmse_q {acc['rmse_q']:.1f} elwr {acc['e_lwr']:.3f} ({time.time() - te:.0f}s)",
                flush=True,
            )
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(
            {"model": net.state_dict(), "ema": ema.state_dict(), "step": step, "epoch": epoch, "cfg": cfg, "mcfg": mcfg}, run / "last.pt"
        )
        if step >= total_steps:
            break
    print("done; best holdout31 objective", best, flush=True)


if __name__ == "__main__":
    main()
