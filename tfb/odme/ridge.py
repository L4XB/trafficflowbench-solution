"""Pure target-count ridge ODME solve (the organizer's baseline objective with the split's own counts and a chosen lambda).

    minimize ||A f - c_split||^2 + lambda * ||f - b_split||^2   s.t. f >= 0

Run: python -m tfb.odme.ridge --lam 5.0 --out results_top3/odme/target_ridge_lam5
Writes <out>/<split>.csv with columns panel, departure_time, path_id, origin_zone, destination_zone, path_flow.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from tfb.common import PANELS, RELEASE
from tfb.odme.baseline_io import load_operator, prior_values, released_counts, released_prior  # organizer baseline readers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=["validation", "private"])
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument(
        "--counts",
        default="released",
        help="released, detector:H0-H1 = the split's mean mainline detector flow over hours H0..H1 (Task-1 data) at the counted links, or blend:H0-H1 = the mean of both",
    )
    ap.add_argument(
        "--prior",
        default="split",
        choices=["split", "mean3", "geo3"],
        help="ridge target: the split's own prior, or the mean / geometric mean of the three released priors",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for split in a.splits:
        frames = []
        for pn in a.panels:
            network = RELEASE / "corridors" / pn / "network"
            path_ids, link_ids, A, paths = load_operator(network)
            if a.prior == "split":
                base = prior_values(RELEASE, pn, split, path_ids)
            else:
                pri = np.stack([prior_values(RELEASE, pn, sp, path_ids) for sp in ("train", "validation", "private")])
                base = pri.mean(0) if a.prior == "mean3" else np.exp(np.log(np.clip(pri, 1e-6, None)).mean(0))
            cf = released_counts(RELEASE, pn, split)
            if a.counts.startswith(("detector:", "blend:")):
                from tfb.common import load_panel

                h0, h1 = (int(v) for v in a.counts.split(":")[1].split("-"))
                pl = load_panel(pn, split)
                pli = {str(l): i for i, l in enumerate(pl.links)}
                fq = np.asarray(pl["flow_m"], np.float32)
                det = [float(np.nanmean(fq[:, h0 * 12 : h1 * 12, pli[l]])) if l in pli else c for l, c in zip(cf.link_id, cf["count"])]
                cf = cf.copy()
                cf["count"] = det if a.counts.startswith("detector:") else [0.5 * (d + c) for d, c in zip(det, cf["count"])]
            counts = cf.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(dtype=float)
            measured = np.array([l in set(cf.link_id) for l in link_ids], dtype=bool)
            aa = np.vstack([A[measured], np.sqrt(a.lam) * np.eye(A.shape[1])])
            bb = np.concatenate([counts[measured], np.sqrt(a.lam) * base])
            f, _ = nnls(aa, bb)
            prior = released_prior(RELEASE, pn, split)
            dep = str(prior.departure_time.iloc[0])
            x = paths[["path_id", "origin_zone", "destination_zone"]].copy()
            x["panel"] = pn
            x["departure_time"] = dep
            x["path_flow"] = f
            frames.append(x[["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]])
            loaded = A[measured] @ f
            s_link = 1 - np.abs(loaded - counts[measured]).sum() / np.abs(counts[measured]).sum()
            print(
                f"{split} {pn}: paths {len(f)} S_link {s_link:.5f} sum {f.sum():.0f} prior sum {base.sum():.0f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        pd.concat(frames, ignore_index=True).to_csv(out / f"{split}.csv", index=False)


if __name__ == "__main__":
    main()
