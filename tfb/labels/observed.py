"""Raw-observation view of the training rows: whether a row's label cell was observed and whether the observation shows a queue.

Used by ``tfb.queue.final --onset-obs-gap0 / --ongoing-obs-gap0``: the label of an observed cell comes from the raw observation
(speed < v_cut) and detector-gap cells get weight 0, because at T+30 the U-Net field that labels the gaps misses fresh queues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tfb.common import CACHE, QUEUE_PANELS, load_panel
from tfb.queue.data import HOR


def cell_info(w: pd.DataFrame, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(observed at T+30, raw observation queued) per onset row."""
    obs = np.zeros(len(keys), bool)
    rq = np.zeros(len(keys), bool)
    for pn in QUEUE_PANELS:
        sp = np.load(CACHE / pn / "train_speed.npy", mmap_mode="r")
        vc = load_panel(pn, "train").param("v_cut")
        widx = np.where(w.panel.to_numpy() == pn)[0]
        sel = np.isin(keys[:, 0], widx)
        if not sel.any():
            continue
        k = keys[sel]
        d = w.d.to_numpy()[k[:, 0]]
        l = k[:, 1]
        t = w["T"].to_numpy()[k[:, 0]] + (
            np.where(k[:, 2] > 0, k[:, 2], HOR) if keys.shape[1] > 2 else HOR
        )  # onset rows carry h = 0 -> T+30
        s = np.asarray(sp[d, t, l])
        obs[sel] = np.isfinite(s)
        rq[sel] = np.isfinite(s) & (s < vc[l])
    return obs, rq
