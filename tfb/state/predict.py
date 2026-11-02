"""Inference for the Task-1 inpainting model.

Holdout predictions (train split, real masks, days = holdout31 + February 2031):
    python -m tfb.state.predict --name unet_s_v1 --holdout
Submission arrays for validation and private:
    python -m tfb.state.predict --name unet_s_v1 --splits validation private
Ensemble several runs / checkpoints (predictions are averaged):
    python -m tfb.state.predict --name ens3 --ckpt results_top3/state/a/best.pt results_top3/state/b/best.pt

Outputs: results_top3/state/<name>/<panel>_<split>.npz with float32 ``speed`` and
``flow`` shaped (days, 288, links), and for --holdout <panel>_holdout.npz with the
additional ``days`` index vector into the train split.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from tfb.common import PANELS
from tfb.state.data import N_FEATURES, february_dates, holdout_dates, load_arrays
from tfb.state.model import build_model
from tfb.state.train import OUT, predict_full


def load_nets(ckpts: list[str], device, use_ema: bool = True) -> list[torch.nn.Module]:
    """Rebuild each checkpoint's UNet (ck["mcfg"]) on `device` with its EMA or raw weights, in eval mode."""
    nets = []
    for c in ckpts:
        ck = torch.load(c, map_location=device)
        net = build_model(ck["mcfg"], N_FEATURES).to(device)
        net.load_state_dict(ck["ema" if use_ema else "model"])
        net.eval()
        nets.append(net)
    return nets


def ensemble_predict(nets, a, days, device, combine: str = "mean"):
    """Average (or take the per-cell median of) the member predictions."""
    vs, qs = [], []
    for net in nets:
        v, q = predict_full(net, a, days, device)
        vs.append(v)
        qs.append(q)
    if combine == "median" and len(nets) > 1:
        return np.median(np.stack(vs), axis=0), np.median(np.stack(qs), axis=0)
    return np.mean(np.stack(vs), axis=0), np.mean(np.stack(qs), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output directory name under results_top3/state")
    ap.add_argument("--ckpt", nargs="*", default=None, help="checkpoints (default: results_top3/state/<name>/best.pt)")
    ap.add_argument("--splits", nargs="*", default=[])
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--raw", action="store_true", help="use the raw (non-EMA) weights")
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--combine", default="mean", choices=["mean", "median"])
    ap.add_argument("--device", default="auto", help="auto (mps if available), mps or cpu")
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    ckpts = args.ckpt or [str(out / "best.pt")]
    nets = load_nets(ckpts, device, use_ema=not args.raw)
    print("checkpoints:", ckpts, flush=True)
    if args.holdout:
        h31 = holdout_dates()
        feb = february_dates()
        hd = sorted(set(h31) | set(feb))
        for panel in args.panels:
            a = load_arrays(panel, "train", with_trans=False)
            days = np.array([a.dates.index(d) for d in hd if d in a.dates])
            pv, pq = ensemble_predict(nets, a, days, device, args.combine)
            np.savez(out / f"{panel}_holdout.npz", speed=pv, flow=pq, days=days)
            print(panel, "holdout", pv.shape, flush=True)
    for split in args.splits:
        for panel in args.panels:
            a = load_arrays(panel, split, with_trans=False)
            days = np.arange(a.n_days)
            pv, pq = ensemble_predict(nets, a, days, device, args.combine)
            assert np.isfinite(pv).all() and np.isfinite(pq).all()
            np.savez(out / f"{panel}_{split}.npz", speed=pv.astype(np.float32), flow=pq.astype(np.float32))
            print(panel, split, pv.shape, "v range", pv.min().round(1), pv.max().round(1), "q min", pq.min().round(1), flush=True)
    (out / "predict_info.json").write_text(json.dumps({"checkpoints": ckpts, "ema": not args.raw, "combine": args.combine}, indent=1))


if __name__ == "__main__":
    main()
