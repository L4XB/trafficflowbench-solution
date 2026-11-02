"""Features of the optional second-stage onset model (``tfb.queue.final --stack-onset``): the window's base probability vector
and its spatial context per link, plus a few first-stage columns. The shipped models do not use the stacker.
"""

from __future__ import annotations

import numpy as np

STACK_KEEP = ("vr_last", "vr_min", "onset_freq", "onset_rate", "clim_h6", "pos", "panel", "fr_sl6", "qlast_el")


def context(p: np.ndarray, keys: np.ndarray, X: np.ndarray, names: list[str]) -> np.ndarray:
    """Per-row stacking features from the window's probability vector (rows of a window are in link order)."""
    out = np.zeros((len(p), 14), np.float32)
    for widx in np.unique(keys[:, 0]):
        sel = np.where(keys[:, 0] == widx)[0]
        q = p[sel]
        L = len(q)
        order = np.argsort(np.argsort(-q))
        pm = q.max()
        ps = q.sum()
        up1 = np.concatenate([[0.0], q[:-1]])
        dn1 = np.concatenate([q[1:], [0.0]])
        up2 = np.concatenate([[0.0, 0.0], q[:-2]])
        dn2 = np.concatenate([q[2:], [0.0, 0.0]])
        out[sel, 0] = q
        out[sel, 1] = up1
        out[sel, 2] = dn1
        out[sel, 3] = up2
        out[sel, 4] = dn2
        out[sel, 5] = pm
        out[sel, 6] = ps
        out[sel, 7] = (q >= 0.35).sum()
        out[sel, 8] = order
        out[sel, 9] = q / max(pm, 1e-6)
        out[sel, 10] = np.maximum(up1, dn1)
        out[sel, 11] = (q >= 0.2).sum()
        out[sel, 12] = np.sort(q)[-2] if L > 1 else 0.0
        out[sel, 13] = float(L)
    return out
