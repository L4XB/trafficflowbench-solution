"""Forecaster with a queue-probability head: a third output channel trained with BCE on (observed speed <= v_cut) at T+5..T+30.

The mean-speed forecast is ambiguous at queue fronts (70 km/h can be half queued at 40 and half free at 100); the head gives the
queue probability directly. Fine-tunes a 2-channel forecast checkpoint: the speed/flow head is copied, the queue channel starts
at the base rate. v_cut = 0.6 x free speed on every link of all ten panels, read from the static input channel vf / 120.
Labels are the raw observation (the loss mask holds observed horizon cells only), so the gap-label bias of the U-Net field
(fresh queues smoothed away) does not enter.

Run: python -m tfb.state.train_qh --name fc_fold0qh --resume results_top3/state/fc_fold0b/best.pt --exclude fold0 \
         --forecast --w_queue 0.5 --minutes 120 --lr 1.5e-4 --warmup 200 --state_loss mse --phys_loss l1 --w_phys 0.03 --eval_every 3
(all train.py options apply; --w_queue weights the BCE term)
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time

import numpy as np
import torch

from tfb.state import train as base
from tfb.state.data import (
    FLOW_SCALE,
    N_FEATURES,
    SPEED_MAX,
    SPEED_SCALE,
    TrainSampler,
    batch_inputs,
    february_dates,
    full_panel_batch,
    holdout_dates,
    load_arrays,
)
from tfb.state.model import build_model

VCUT_RATIO = 0.6
BASE_RATE = 0.12  # share of queued horizon cells in the forecast windows (initial head bias)


def parse() -> argparse.Namespace:
    """The tfb.state.train command line plus --w_queue (weight of the BCE term), which is stripped from sys.argv first."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--w_queue", type=float, default=0.5)
    extra, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = base.parse()
    args.w_queue = extra.w_queue
    return args


def queue_loss(out: torch.Tensor, s: dict) -> torch.Tensor:
    """Mean BCE of the queue logit on the loss cells; label = observed speed <= 0.6 x free speed."""
    m = s["loss_mask"]
    B, _T, W = m.shape
    vcut = (VCUT_RATIO * 120.0 * s["static"][..., 1]).view(B, 1, W)  # static[..., 1] = vf / 120
    y = (torch.nan_to_num(s["v"], nan=1e3) <= vcut).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 2], y, reduction="none")
    return (bce * m).sum() / m.sum().clamp(min=1)


def predict_full_qh(net: torch.nn.Module, a, days: np.ndarray, device, chunk: int = 8):
    """(len(days), 288, L) speed, flow and queue probability with the real masks in ``a``."""
    net.eval()
    pv, pq, pp = [], [], []
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
            pp.append(torch.sigmoid(out[:, 2, :, :L]).float().cpu().numpy())
    net.train()
    return np.concatenate(pv).astype(np.float32), np.concatenate(pq).astype(np.float32), np.concatenate(pp).astype(np.float32)


def evaluate_forecast_qh(net, panels, day_sets, device, origins=(96, 120, 144, 168, 192, 216)) -> dict:
    """Holdout forecast quality: speed RMSE, queue IoU of the speed threshold and of the head (p >= 0.5), head BCE."""
    res = {}
    for hname, dmap in day_sets.items():
        se = n = 0.0
        iv = uv = ip = up = 0
        bce = 0.0
        nb = 0
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
            pv, _, pp = predict_full_qh(net, b, np.arange(len(dd)), device)
            for i, T in enumerate(TT):
                tr = truth[i, T + 1 : T + 7]
                fin = np.isfinite(tr)
                prv = pv[i, T + 1 : T + 7]
                prp = pp[i, T + 1 : T + 7]
                se += float(((prv - tr)[fin] ** 2).sum())
                n += int(fin.sum())
                qt = fin & (tr <= vc[None, :])
                qv = prv <= vc[None, :]
                qp = prp >= 0.5
                iv += int((qt & qv).sum())
                uv += int((qt | qv).sum())
                ip += int((qt & qp).sum())
                up += int((qt | qp).sum())
                pc = np.clip(prp[fin], 1e-6, 1 - 1e-6)
                yt = qt[fin]
                bce += float(-(yt * np.log(pc) + (~yt) * np.log(1 - pc)).sum())
                nb += int(fin.sum())
        rmse = (se / max(n, 1)) ** 0.5
        iou_v = iv / max(uv, 1)
        iou_p = ip / max(up, 1)
        res[hname] = {
            "S_state": iou_p,
            "S_physics": rmse,
            "objective": 0.5 * (iou_v + iou_p) - rmse / 100.0,
            "fc_rmse_v": rmse,
            "fc_queue_iou": iou_v,
            "head_iou": iou_p,
            "head_bce": bce / max(nb, 1),
            "per_panel": {},
        }
    return res


def adapt_head(sd: dict) -> dict:
    """2-channel state dict -> 3-channel: speed/flow head copied, queue channel zero weights and base-rate bias."""
    sd = dict(sd)
    w, b = sd["head.weight"], sd["head.bias"]
    if w.shape[0] == 2:
        w3 = torch.zeros((3,) + tuple(w.shape[1:]), dtype=w.dtype, device=w.device)
        w3[:2] = w
        b3 = torch.zeros(3, dtype=b.dtype, device=b.device)
        b3[:2] = b
        b3[2] = math.log(BASE_RATE / (1 - BASE_RATE))
        sd["head.weight"], sd["head.bias"] = w3, b3
    return sd


def main() -> None:
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import os

    device = torch.device(os.environ.get("TRAIN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
    run = base.OUT / args.name
    run.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["features"] = N_FEATURES
    cfg["queue_head"] = True
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(run / "log.jsonl", "a")  # noqa: SIM115  (appended to once per epoch for the whole run)
    t0 = time.time()
    panels = [load_arrays(p, "train") for p in args.panels]
    h31 = holdout_dates()
    feb = february_dates()
    folds = {f"fold{k}": {d for i, d in enumerate(panels[0].dates) if i % 4 == k} for k in range(4)}
    excl = {"both": set(h31) | set(feb), "holdout31": set(h31), "feb": set(feb), "none": set(), **folds}[args.exclude]
    train_days = {a.name: np.array([i for i, d in enumerate(a.dates) if d not in excl]) for a in panels}
    day_sets = {
        "holdout31": {a.name: np.array([a.dates.index(d) for d in h31 if d in a.dates]) for a in panels},
        "feb2031": {a.name: np.array([a.dates.index(d) for d in feb if d in a.dates]) for a in panels},
    }
    print(
        f"loaded {len(panels)} panels in {time.time() - t0:.0f}s; train days/panel {len(train_days[panels[0].name])}; excluded {len(excl)} dates",
        flush=True,
    )
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
    )
    w = np.array([max(a.n_links, args.crop) ** args.panel_power for a in panels])
    sampler.pw = w / w.sum()
    mcfg = {"widths": args.widths, "blocks": args.blocks, "bottleneck_dilations": args.dilations, "out_ch": 3}
    mcfg["res_scale"] = [args.res_scale_v, args.res_scale_q]
    net = build_model(mcfg, N_FEATURES).to(device)
    net.res_scale = (args.res_scale_v, args.res_scale_q)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(adapt_head(ck["model"]))
        ema.load_state_dict(adapt_head(ck["ema"]))
        print("resumed from", args.resume, "(queue head added)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))
    total_steps = None if args.minutes > 0 else args.epochs * args.steps_per_epoch
    step = 0
    best = -1.0

    def lr_at(st: int, tot: int) -> float:
        """Learning rate at step st of tot: linear warmup over --warmup steps, then cosine decay from --lr to 1e-5."""
        if st < args.warmup:
            return args.lr * (st + 1) / args.warmup
        pr = min(1.0, (st - args.warmup) / max(1, tot - args.warmup))
        return 1e-5 + (args.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * pr))

    t_train0 = time.time()
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        acc = {"loss": 0.0, "rmse_v": 0.0, "rmse_q": 0.0, "e_lwr": 0.0, "bce": 0.0}
        for _ in range(args.steps_per_epoch):
            tot = total_steps if total_steps else max(step + 1, args.steps_per_epoch * 2)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tot)
            s = sampler.sample()
            x, v_lin, ql_lin = batch_inputs(s)
            args._step = step
            out = net(x)
            loss0, parts = base.loss_fn(out, v_lin, ql_lin, s, args)
            lq = queue_loss(out, s)
            loss = loss0 + args.w_queue * lq
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
            acc["bce"] += lq.item()
            for k in ("rmse_v", "rmse_q", "e_lwr"):
                acc[k] += parts[k]
            if not math.isfinite(loss.item()) or loss.item() == 0.0:
                print("ABORT: loss", loss.item(), "at step", step, flush=True)
                return
        for k in acc:
            acc[k] /= args.steps_per_epoch
        train_min = (time.time() - t_train0) / 60
        if args.minutes > 0 and total_steps is None:
            total_steps = int(args.minutes / ((time.time() - te) / 60)) * args.steps_per_epoch
            print(f"time budget {args.minutes} min -> {total_steps // args.steps_per_epoch} epochs", flush=True)
        rec = {"epoch": epoch, "step": step, "train": acc, "lr": opt.param_groups[0]["lr"], "min": round(train_min, 1)}
        if epoch % args.eval_every == 0 or step >= total_steps:
            ev = evaluate_forecast_qh(ema, panels, day_sets, device)
            rec["eval"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_panel"} for k, v in ev.items()}
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
            h = ev["holdout31"]
            print(
                f"ep {epoch:3d} step {step:6d} {train_min:5.1f}min loss {acc['loss']:.4f} rmse_v {acc['rmse_v']:.3f} bce {acc['bce']:.4f} | h31 "
                f"iou_v {h['fc_queue_iou']:.4f} iou_head {h['head_iou']:.4f} bce {h['head_bce']:.4f} rmse {h['fc_rmse_v']:.3f} obj {obj:.5f}"
                f"{' *' if rec.get('saved') else ''}  ({time.time() - te:.0f}s)",
                flush=True,
            )
        else:
            print(
                f"ep {epoch:3d} step {step:6d} {train_min:5.1f}min loss {acc['loss']:.4f} rmse_v {acc['rmse_v']:.3f} bce {acc['bce']:.4f} "
                f"({time.time() - te:.0f}s)",
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
