"""Per-link feature builder for the queue models (history strictly before T)."""

from __future__ import annotations

import os
from functools import cache

import numpy as np

from tfb.common import STEPS
from tfb.queue.data import HIST, HOR


def shift(a: np.ndarray, k: int, fill: float = np.nan) -> np.ndarray:
    """Shift along the last axis (link order upstream->downstream): k>0 takes the value k links upstream."""
    out = np.full_like(a, fill, dtype=np.float64)
    if k > 0:
        out[..., k:] = a[..., :-k]
    elif k < 0:
        out[..., :k] = a[..., -k:]
    else:
        out[...] = a
    return out


@cache
def link_direction(panel: str) -> int:
    """+1 when the cache link index increases downstream (E/N corridors), -1 when it increases upstream (W/S corridors).

    Read from the ordered link sequences of path_set.csv; the cache order is spatial in both cases (Spearman +-1 on every panel).
    """
    import pandas as pd

    from tfb.common import RELEASE, load_panel

    pos = {str(l): i for i, l in enumerate(load_panel(panel, "train").links)}
    votes = 0
    for seq in pd.read_csv(RELEASE / "corridors" / panel / "network" / "path_set.csv").link_seq:
        idx = [pos[l] for l in seq.split(";") if l in pos]
        if len(idx) > 1:
            votes += int(np.sign(np.diff(idx)).sum())
    return 1 if votes >= 0 else -1


def _dir(pq) -> int:
    """Neighbour direction for the feature builders: link_direction when QUEUE_DIRFIX=1, else the legacy index order (+1)."""
    return link_direction(pq.panel) if os.environ.get("QUEUE_DIRFIX") == "1" else 1


class Climatology:
    """Truth queue frequencies over each panel's fit days: freq, freq_wk, freq_wd (288, L) over all, weekend and weekday days.

    onset_freq (24, L): queued share at T+HOR of the onset windows per hour of T+HOR (+-1 h pooled); onset_rate (L,): all hours.
    """

    def __init__(self, pqs: dict, fit_days: dict[str, np.ndarray], onset_windows=None):
        self.freq, self.onset_freq, self.onset_rate = {}, {}, {}
        for name, pq in pqs.items():
            days = fit_days[name]
            tr = np.asarray(pq.truth[days])  # (n, 288, L)
            self.freq[name] = tr.mean(0)  # (288, L)
            # weekday / weekend split
            wd = np.array([np.datetime64(pq.dates[d]).astype("datetime64[D]").astype(int) % 7 for d in days])
            # numpy day 0 = Thursday 1970-01-01 -> weekend when (x+3)%7 >= 5
            wk = ((wd + 3) % 7) >= 5
            self.freq_wk = getattr(self, "freq_wk", {})
            self.freq_wd = getattr(self, "freq_wd", {})
            self.freq_wk[name] = tr[wk].mean(0) if wk.any() else self.freq[name]
            self.freq_wd[name] = tr[~wk].mean(0) if (~wk).any() else self.freq[name]
            L = pq.p.n_links
            of = np.zeros((24, L))
            cnt = np.zeros(24)
            if onset_windows is not None:
                ow = onset_windows[(onset_windows.panel == name) & onset_windows.d.isin(set(days.tolist()))]
                for r in ow.itertuples():
                    hr = ((r.T + HOR) // 12) % 24
                    of[hr] += pq.truth[r.d, r.T + HOR]
                    cnt[hr] += 1
            # smooth over +-1 hour
            ofs = np.zeros_like(of)
            cs = np.zeros_like(cnt)
            for h in range(24):
                for k in (-1, 0, 1):
                    ofs[h] += of[(h + k) % 24]
                    cs[h] += cnt[(h + k) % 24]
            self.onset_freq[name] = ofs / np.maximum(cs, 1)[:, None]
            self.onset_rate[name] = of.sum(0) / max(cnt.sum(), 1)


def static_feats(pq) -> tuple[np.ndarray, list[str]]:
    """(L, F) per-link statics: lanes, per-lane capacity and jam density, free speed, length, position, ramp flags, neighbour ratios."""
    dr = _dir(pq)
    sh = lambda a, k, fill=np.nan: shift(a, dr * k, fill)  # direction-consistent neighbours (QUEUE_DIRFIX)
    p = pq.p
    L = p.n_links
    lanes = p.param("lanes")
    cap = p.param("capacity")
    vf = p.param("free_speed")
    ln = p.param("length_km")
    cap_up = sh(cap, 1)
    cap_dn = sh(cap, -1)
    on_att = np.asarray(p.meta["params"]["on_attached"], float)
    off_att = np.asarray(p.meta["params"]["off_attached"], float)
    cols = {
        "lanes": lanes,
        "cap_lane": cap / lanes,
        "vf": vf,
        "length": ln,
        "pos": np.arange(L) / L,
        "cap_ratio_up": cap / np.where(np.isfinite(cap_up), cap_up, cap),
        "cap_ratio_dn": np.where(np.isfinite(cap_dn), cap_dn, cap) / cap,
        "on_att": on_att,
        "off_att": off_att,
        "on_att_up1": sh(on_att, 1, 0),
        "on_att_dn1": sh(on_att, -1, 0),
        "kjam_lane": p.param("k_jam") / lanes,
    }
    return np.stack(list(cols.values()), 1), list(cols)


def ramp_series(pq, d: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(HIST, L) on- and off-ramp flow attached per link over the history (NaN-> 0, missing ramps ignored)."""
    rf = np.asarray(pq.p["ramp_flow"][d, T - HIST : T])
    rl = np.asarray(pq.p.meta["ramps"]["ramp_link"])
    rt = np.asarray(pq.p.meta["ramps"]["ramp_type"])
    L = pq.p.n_links
    on = np.zeros((HIST, L))
    off = np.zeros((HIST, L))
    v = np.where(np.isfinite(rf), rf, 0.0)
    for i in range(len(rl)):
        if rl[i] < 0:
            continue
        (on if rt[i] == "OR" else off)[:, rl[i]] += v[:, i]
    return on, off


def link_feats(pq, d: int, T: int, clim: Climatology, static: np.ndarray, static_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """(L, F) features from the visible history T-60..T-5 plus climatology and statics."""
    dr = _dir(pq)
    sh = lambda a, k, fill=np.nan: shift(a, dr * k, fill)  # direction-consistent neighbours (QUEUE_DIRFIX)
    p = pq.p
    L = p.n_links
    sp = np.asarray(pq.speed[d, T - HIST : T], float)
    fl = np.asarray(pq.flow[d, T - HIST : T], float)
    oc = np.asarray(pq.occ[d, T - HIST : T], float)
    q = np.asarray(pq.obs[d, T - HIST : T])
    vr = sp / pq.vcut[None, :]  # speed / v_cut
    fr = fl / pq.cap[None, :]  # flow / capacity
    kr = (fl / np.maximum(sp, 1.0)) / p.param("k_crit")[None, :]

    def last_valid(a):
        """(L,) latest finite value of each link over the HIST history steps (NaN if none)."""
        out = np.full(L, np.nan)
        for t in range(HIST - 1, -1, -1):
            m = np.isnan(out) & np.isfinite(a[t])
            out[m] = a[t][m]
        return out

    def nanmean_last(a, n):
        """(L,) NaN-mean over the last n history steps."""
        return np.nanmean(a[-n:], 0)

    with np.errstate(all="ignore"):
        vr_last = last_valid(vr)
        fr_last = last_valid(fr)
        oc_last = last_valid(oc)
        kr_last = last_valid(kr)
        vr_m3 = nanmean_last(vr, 3)
        vr_m6 = nanmean_last(vr, 6)
        vr_m12 = np.nanmean(vr, 0)
        vr_min = np.nanmin(vr, 0)
        fr_m3 = nanmean_last(fr, 3)
        fr_m12 = np.nanmean(fr, 0)
        fr_max = np.nanmax(fr, 0)
        kr_m3 = nanmean_last(kr, 3)
        kr_max = np.nanmax(kr, 0)
        # slopes over the last 6 steps (least squares on available points)
        x = np.arange(HIST, dtype=float)

        def slope(a, n):
            """(L,) least-squares slope per step over the last n steps (finite points only); 0 with fewer than 3 points."""
            aa = a[-n:]
            xx = x[-n:]
            m = np.isfinite(aa)
            cnt = m.sum(0)
            xm = np.where(m, xx[:, None], 0).sum(0) / np.maximum(cnt, 1)
            ym = np.where(m, aa, 0).sum(0) / np.maximum(cnt, 1)
            num = (np.where(m, (xx[:, None] - xm) * (aa - ym), 0)).sum(0)
            den = (np.where(m, (xx[:, None] - xm) ** 2, 0)).sum(0)
            return np.where(cnt >= 3, num / np.maximum(den, 1e-9), 0.0)

        vr_sl6 = slope(vr, 6)
        vr_sl12 = slope(vr, 12)
        fr_sl6 = slope(fr, 6)
    qcnt = q.sum(0).astype(float)
    qlast = q[-1].astype(float)
    # steps since last queued observation (12 = never)
    since = np.full(L, 12.0)
    for t in range(HIST):
        since[q[t]] = HIST - 1 - t
    qlast_el = pq.last_eligible_flag(d, T).astype(float)
    on, off = ramp_series(pq, d, T)
    on_last = on[-1] / pq.cap
    on_m3 = on[-3:].mean(0) / pq.cap
    off_last = off[-1] / pq.cap
    on_up = sh(on_last, 1, 0) + sh(on_last, 2, 0)
    # queue geometry at the last step
    comp_len = np.zeros(L)
    dist_tail = np.full(L, 99.0)
    dist_head = np.full(L, 99.0)
    qi = np.where(q[-1])[0]
    if len(qi):
        # connected components in link order
        runs = np.split(qi, np.where(np.diff(qi) > 1)[0] + 1)
        for run in runs:
            comp_len[run] = len(run)
        # distance to nearest queued link upstream (tail side: queued links downstream of l) / downstream
        for l in range(L):
            dn = qi[qi >= l]
            up = qi[qi <= l]
            if len(dn):
                dist_head[l] = dn.min() - l  # nearest queued link at or downstream
            if len(up):
                dist_tail[l] = l - up.max()  # nearest queued link at or upstream
    slot = (T + HOR) % STEPS
    hr = (slot // 12) % 24
    wd = (np.datetime64(pq.dates[d]).astype("datetime64[D]").astype(int) + 3) % 7
    wk = float(wd >= 5)
    f = clim.freq[pq.panel]
    fwk = (clim.freq_wk if wk else clim.freq_wd)[pq.panel]
    cols = {
        "vr_last": vr_last,
        "vr_m3": vr_m3,
        "vr_m6": vr_m6,
        "vr_m12": vr_m12,
        "vr_min": vr_min,
        "vr_sl6": vr_sl6,
        "vr_sl12": vr_sl12,
        "fr_last": fr_last,
        "fr_m3": fr_m3,
        "fr_m12": fr_m12,
        "fr_max": fr_max,
        "fr_sl6": fr_sl6,
        "oc_last": oc_last,
        "kr_last": kr_last,
        "kr_m3": kr_m3,
        "kr_max": kr_max,
        "qcnt": qcnt,
        "qlast": qlast,
        "qlast_el": qlast_el,
        "since": since,
        "q_up1": sh(qlast, 1, 0),
        "q_up2": sh(qlast, 2, 0),
        "q_up3": sh(qlast, 3, 0),
        "q_dn1": sh(qlast, -1, 0),
        "q_dn2": sh(qlast, -2, 0),
        "q_dn3": sh(qlast, -3, 0),
        "qcnt_up1": sh(qcnt, 1, 0),
        "qcnt_dn1": sh(qcnt, -1, 0),
        "vr_up1": sh(vr_last, 1),
        "vr_dn1": sh(vr_last, -1),
        "vr_up2": sh(vr_last, 2),
        "vr_dn2": sh(vr_last, -2),
        "fr_up1": sh(fr_last, 1),
        "fr_dn1": sh(fr_last, -1),
        "fr_dn2": sh(fr_last, -2),
        "vr_m3_up1": sh(vr_m3, 1),
        "vr_m3_dn1": sh(vr_m3, -1),
        "vr_min_nb": np.fmin(np.fmin(sh(vr_last, 1), sh(vr_last, -1)), vr_last),
        "fr_max_nb3": np.fmax.reduce([sh(fr_last, k) for k in (-3, -2, -1, 0, 1, 2, 3)]),
        "on_last": on_last,
        "on_m3": on_m3,
        "off_last": off_last,
        "on_up": on_up,
        "comp_len": comp_len,
        "dist_tail": np.minimum(dist_tail, 30),
        "dist_head": np.minimum(dist_head, 30),
        "nq_last": np.full(L, float(len(qi))),
        "nq_hist_links": np.full(L, float((qcnt > 0).sum())),
        "hour": np.full(L, float(hr)),
        "slot_sin": np.full(L, np.sin(2 * np.pi * slot / STEPS)),
        "slot_cos": np.full(L, np.cos(2 * np.pi * slot / STEPS)),
        "weekend": np.full(L, wk),
        "wd": np.full(L, float(wd)),
        "clim_h6": f[(T + 6) % STEPS],
        "clim_h3": f[(T + 3) % STEPS],
        "clim_h1": f[(T + 1) % STEPS],
        "clim_h9": f[(T + 9) % STEPS],
        "clim_h6_max3": np.max(f[[(T + 5) % STEPS, (T + 6) % STEPS, (T + 7) % STEPS]], 0),
        "clim_wk_h6": fwk[(T + 6) % STEPS],
        "clim_day_max": f.max(0),
        "clim_h6_up1": sh(f[(T + 6) % STEPS], 1, 0),
        "clim_h6_dn1": sh(f[(T + 6) % STEPS], -1, 0),
        "onset_freq": clim.onset_freq[pq.panel][hr],
        "onset_rate": clim.onset_rate[pq.panel],
        "onset_freq_up1": sh(clim.onset_freq[pq.panel][hr], 1, 0),
        "onset_freq_dn1": sh(clim.onset_freq[pq.panel][hr], -1, 0),
    }
    names = list(cols) + static_names
    X = np.concatenate([np.stack(list(cols.values()), 1), static], 1)
    return X, names


def horizon_clim(pq, T: int, clim: Climatology) -> np.ndarray:
    """(HOR, L) climatology at each horizon step."""
    f = clim.freq[pq.panel]
    return np.stack([f[(T + h) % STEPS] for h in range(1, HOR + 1)])


# ---------------------------------------------------------------------------
# v2 additions: the partially visible origin step T, and history trend features
# ---------------------------------------------------------------------------
def origin_feats(pq, d: int, T: int, rng: np.random.Generator | None, drop_rate: float | None) -> tuple[np.ndarray, list[str]]:
    """Features from the observation AT the origin T (masked layer in the evaluation splits).

    In training (unmasked layer) a random share `drop_rate` of the links is hidden to mimic the
    Task-1 mask that applies at T in the evaluation splits.
    """
    dr = _dir(pq)
    sh = lambda a, k, fill=np.nan: shift(a, dr * k, fill)  # direction-consistent neighbours (QUEUE_DIRFIX)
    L = pq.p.n_links
    sp = np.asarray(pq.speed[d, T], float).copy()
    fl = np.asarray(pq.flow[d, T], float).copy()
    oc = np.asarray(pq.occ[d, T], float).copy()
    if drop_rate is not None and rng is not None:
        hide = rng.random(L) < drop_rate
        sp[hide] = np.nan
        fl[hide] = np.nan
        oc[hide] = np.nan
    vr = sp / pq.vcut
    fr = fl / pq.cap
    q = np.where(np.isfinite(sp), (sp <= pq.vcut).astype(float), np.nan)
    vis = np.isfinite(sp).astype(float)
    cols = {
        "vrT": vr,
        "frT": fr,
        "ocT": oc,
        "qT": q,
        "visT": vis,
        "vrT_up1": sh(vr, 1),
        "vrT_dn1": sh(vr, -1),
        "qT_up1": sh(q, 1),
        "qT_dn1": sh(q, -1),
        "vrT_min_nb": np.fmin(np.fmin(sh(vr, 1), sh(vr, -1)), vr),
    }
    return np.stack(list(cols.values()), 1), list(cols)


def trend_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """History-wide queue geometry trends (same value for every link of the window)."""
    q = np.asarray(pq.obs[d, T - HIST : T])
    L = pq.p.n_links
    nq = q.sum(1).astype(float)  # queued links per step
    tail = np.array([np.where(q[t])[0].min() if q[t].any() else np.nan for t in range(HIST)], float)
    head = np.array([np.where(q[t])[0].max() if q[t].any() else np.nan for t in range(HIST)], float)

    def sl(a):
        """Least-squares slope per step over the finite entries of a; 0.0 with fewer than 3."""
        m = np.isfinite(a)
        if m.sum() < 3:
            return 0.0
        x = np.arange(HIST)[m]
        y = a[m]
        return float(np.polyfit(x, y, 1)[0])

    vals = {
        "nq_sl": sl(nq),
        "nq_last3_minus_first3": nq[-3:].mean() - nq[:3].mean(),
        "nq_max": nq.max(),
        "nq_mean": nq.mean(),
        "tail_sl": sl(tail),
        "head_sl": sl(head),
        "tail_last": tail[-1] if np.isfinite(tail[-1]) else -1.0,
        "head_last": head[-1] if np.isfinite(head[-1]) else -1.0,
        "n_steps_queued": float((nq > 0).sum()),
    }
    idx = np.arange(L, dtype=float)
    per_link = {
        "rel_tail": idx - (tail[-1] if np.isfinite(tail[-1]) else -99),
        "rel_head": (head[-1] if np.isfinite(head[-1]) else -99) - idx,
    }
    X = np.concatenate([np.tile(np.array(list(vals.values()))[None, :], (L, 1)), np.stack(list(per_link.values()), 1)], 1)
    return X, list(vals) + list(per_link)


# ---------------------------------------------------------------------------
# v3 additions: same-day context before the 60-minute history (legal: timestamps <= T)
# ---------------------------------------------------------------------------
def dayctx_feats(pq, d: int, T: int, rng: np.random.Generator | None, drop_rate: float | None) -> tuple[np.ndarray, list[str]]:
    """Features from the day's observations at steps 0..T-13 (before the released history).

    In the evaluation splits these steps carry the Task-1 mask; in training a random share
    `drop_rate` of the cells is hidden to mimic it.
    """
    L = pq.p.n_links
    t_end = T - HIST
    if t_end <= 0:
        z = np.zeros(L)
        cols = {
            "age_q": z,
            "frac_q_day": z,
            "q_prev_hour": z,
            "day_min_vr": np.ones(L) * 2,
            "day_max_fr": z,
            "nq_day_max": z,
            "since_any_q": np.full(L, 300.0),
            "any_q_day": z,
            "n_vis_day": z,
        }
        return np.stack(list(cols.values()), 1), list(cols)
    # raw observations only: the gap fills of the evaluation splits come from the offline (whole-day) state, i.e. from after T
    raw_sp, raw_fl = (pq.p["speed"], pq.p["flow"]) if pq.split == "train" else (pq.p["speed_m"], pq.p["flow_m"])
    sp = np.asarray(raw_sp[d, :t_end], float).copy()
    fl = np.asarray(raw_fl[d, :t_end], float).copy()
    if drop_rate is not None and rng is not None:
        hide = rng.random(sp.shape) < drop_rate
        sp[hide] = np.nan
        fl[hide] = np.nan
    vis = np.isfinite(sp)
    q = vis & (sp <= pq.vcut[None, :])
    with np.errstate(all="ignore"):
        vr = sp / pq.vcut[None, :]
        fr = fl / pq.cap[None, :]
        day_min_vr = np.where(vis.any(0), np.nanmin(np.where(vis, vr, np.nan), 0), 2.0)
        day_max_fr = np.where(vis.any(0), np.nanmax(np.where(vis, fr, np.nan), 0), 0.0)
    first_q = np.array([np.where(q[:, l])[0].min() if q[:, l].any() else -1 for l in range(L)], float)
    age_q = np.where(first_q >= 0, T - first_q, 0.0)
    frac_q_day = q.sum(0) / np.maximum(vis.sum(0), 1)
    q_prev_hour = q[max(0, t_end - 12) : t_end].any(0).astype(float)
    nq = q.sum(1)
    nq_day_max = float(nq.max())
    last_any = np.where(nq > 0)[0]
    since_any = float(T - last_any.max()) if len(last_any) else 300.0
    cols = {
        "age_q": age_q,
        "frac_q_day": frac_q_day,
        "q_prev_hour": q_prev_hour,
        "day_min_vr": day_min_vr,
        "day_max_fr": day_max_fr,
        "nq_day_max": np.full(L, nq_day_max),
        "since_any_q": np.full(L, since_any),
        "any_q_day": np.full(L, float(len(last_any) > 0)),
        "n_vis_day": vis.mean(0),
    }
    return np.stack(list(cols.values()), 1), list(cols)


# ---------------------------------------------------------------------------
# v4 additions: queue-balance features (arrival vs discharge around each queue component)
# ---------------------------------------------------------------------------
def balance_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Per link: flow balance of the queue component it belongs to (or the nearest one) at the last steps."""
    L = pq.p.n_links
    cap = pq.cap
    q = np.asarray(pq.obs[d, T - HIST : T])
    fl = np.asarray(pq.flow[d, T - HIST : T], float)
    sp = np.asarray(pq.speed[d, T - HIST : T], float)
    with np.errstate(all="ignore"):
        f3 = np.nanmean(fl[-3:], 0)
        f6 = np.nanmean(fl[-6:], 0)
        s3 = np.nanmean(sp[-3:], 0)
    f3 = np.where(np.isfinite(f3), f3, np.nan)
    f6 = np.where(np.isfinite(f6), f6, np.nan)
    last = q[-1]
    qi = np.where(last)[0]
    comp_id = np.full(L, -1)
    comps = []
    if len(qi):
        for k, run in enumerate(np.split(qi, np.where(np.diff(qi) > 1)[0] + 1)):
            comp_id[run] = k
            comps.append((run.min(), run.max()))
    q_in = np.full(L, np.nan)
    q_out = np.full(L, np.nan)
    q_in6 = np.full(L, np.nan)
    q_out6 = np.full(L, np.nan)
    disch = np.full(L, np.nan)
    own_comp = np.zeros(L)
    comp_len = np.zeros(L)
    tail_speed = np.full(L, np.nan)
    if comps:
        idx = np.arange(L)
        # nearest component for every link
        centers = np.array([(a + b) / 2 for a, b in comps])
        near = np.argmin(np.abs(idx[:, None] - centers[None, :]), 1)
        for l in range(L):
            k = comp_id[l] if comp_id[l] >= 0 else near[l]
            a, b = comps[k]
            own_comp[l] = float(comp_id[l] >= 0)
            comp_len[l] = b - a + 1
            up = a - 1
            dn = b + 1
            q_in[l] = f3[up] if up >= 0 else np.nan
            q_in6[l] = f6[up] if up >= 0 else np.nan
            q_out[l] = f3[dn] if dn < L else np.nan
            q_out6[l] = f6[dn] if dn < L else np.nan
            disch[l] = f3[b] / cap[b]
            tail_speed[l] = s3[a] / pq.vcut[a]
    cols = {
        "bal_in": q_in / cap,
        "bal_out": q_out / cap,
        "bal_diff": (q_in - q_out) / cap,
        "bal_diff6": (q_in6 - q_out6) / cap,
        "bal_disch": disch,
        "bal_own": own_comp,
        "bal_comp_len": comp_len,
        "bal_tail_vr": tail_speed,
    }
    return np.stack(list(cols.values()), 1), list(cols)


# ---------------------------------------------------------------------------
# v5 additions: absolute speeds and the link's own cut relative to its neighbours
# ---------------------------------------------------------------------------
def abs_speed_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """(L, F) absolute history speeds of each link and its neighbours, its v_cut, and that cut against the neighbours' cuts and speeds."""
    dr = _dir(pq)
    sh = lambda a, k, fill=np.nan: shift(a, dr * k, fill)  # direction-consistent neighbours (QUEUE_DIRFIX)
    sp = np.asarray(pq.speed[d, T - HIST : T], float)
    L = pq.p.n_links
    vc = pq.vcut
    with np.errstate(all="ignore"):
        v_last = np.full(L, np.nan)
        for t in range(HIST - 1, -1, -1):
            m = np.isnan(v_last) & np.isfinite(sp[t])
            v_last[m] = sp[t][m]
        v_m3 = np.nanmean(sp[-3:], 0)
        v_min = np.nanmin(sp, 0)
    nb_v = np.nanmean(np.stack([sh(v_last, k) for k in (-2, -1, 1, 2)]), 0)
    cols = {
        "v_last": v_last,
        "v_m3": v_m3,
        "v_min": v_min,
        "v_up1": sh(v_last, 1),
        "v_dn1": sh(v_last, -1),
        "v_nb_mean": nb_v,
        "vcut": vc,
        "vcut_minus_up1": vc - sh(vc, 1, np.nan),
        "vcut_minus_dn1": vc - sh(vc, -1, np.nan),
        "vcut_minus_nbmean": vc - np.nanmean(np.stack([sh(vc, k, np.nan) for k in (-2, -1, 1, 2)]), 0),
        "nb_v_minus_vcut": nb_v - vc,
        "v_last_minus_vcut": v_last - vc,
    }
    return np.stack(list(cols.values()), 1), list(cols)


# ---------------------------------------------------------------------------
# v6 additions: in-month recurrence from the same split's earlier days (legal: timestamps <= T)
# ---------------------------------------------------------------------------
_BOARDLIKE = {}


def boardlike_windows(panel: str) -> dict:
    """Per day: list of origins T of the evaluation-like windows (first five per condition and month, merged greedy selection)."""
    if panel in _BOARDLIKE:
        return _BOARDLIKE[panel]

    import pandas as pd

    from tfb.common import PROJECT, load_panel
    from tfb.labels.selector import merged_greedy

    p = load_panel(panel, "train")
    f = PROJECT / "results_top3" / "queue" / f"validity_{panel}_dn.parquet"
    out = {}
    if f.exists():
        v = pd.read_parquet(f)
        v["month"] = [p.dates[d][:7] for d in v.d]
        for m, vm in v.groupby("month"):
            for r in merged_greedy(vm.reset_index(drop=True), quota=5).itertuples():
                out.setdefault(int(r.d), []).append(int(r.T))
    _BOARDLIKE[panel] = out
    return out


def inmonth_feats(pq, d: int, T: int, rng: np.random.Generator | None, drop_rate: float | None, cond: str) -> tuple[np.ndarray, list[str]]:
    """Queue recurrence at this time of day over the earlier days of the same month/split, per link."""
    dr = _dir(pq)
    sh = lambda a, k, fill=np.nan: shift(a, dr * k, fill)  # direction-consistent neighbours (QUEUE_DIRFIX)
    L = pq.p.n_links
    vc = pq.vcut
    month = pq.dates[d][:7]
    import os

    mode = os.environ.get("QUEUE_INM_ROLL", "")
    roll = mode == "1" or (mode == "eval" and pq.split == "private")  # eval: only the private split borrows (the earlier evaluation month)
    if roll:
        # rolling: the 16 days before d whatever the month -- the scored windows sit on days 1-4 of their month (70 % of them),
        # where the in-month window is empty; evaluation splits borrow the previous split's last days (pq.prev_speed, legal)
        nroll = int(os.environ.get("QUEUE_INM_ROLL_DAYS", "16"))
        if pq.split == "train":
            earlier = list(range(max(0, d - nroll), d))
            src = [(pq.speed, k) for k in earlier]
        else:
            own = list(range(max(0, d - nroll), d))
            need = nroll - len(own)
            prev = getattr(pq, "prev_speed", None)
            src = ([(prev, k) for k in range(prev.shape[0] - need, prev.shape[0])] if prev is not None and need > 0 else []) + [
                (pq.speed, k) for k in own
            ]
            earlier = [k for _, k in src]
    else:
        earlier = [k for k in range(d) if pq.dates[k][:7] == month] if pq.split == "train" else list(range(d))
        earlier = earlier[-16:]  # evaluation windows never have more than ~16 earlier days
        ecap = os.environ.get(
            "QUEUE_INM_EVALCAP"
        )  # diagnostics: evaluation windows see at most this many earlier days (mimics month start)
        if ecap is not None and pq.split != "train":
            earlier = earlier[-int(ecap) :] if int(ecap) > 0 else []
        cap = os.environ.get("INMONTH_CAP")  # training-time augmentation: scored evaluation windows have 0-11 earlier days
        if cap is not None and pq.split == "train" and rng is not None and rng.random() < float(os.environ.get("INMONTH_CAP_P", "1.0")):
            k = int(rng.integers(0, int(cap) + 1))
            earlier = earlier[-k:] if k > 0 else []
        src = [(pq.speed, k) for k in earlier]
    c = T + HOR if cond == "onset" else T + 3
    lo, hi = max(0, c - 6), min(STEPS, c + 7)
    lo2, hi2 = max(0, c - 2), min(STEPS, c + 3)
    q_sum = np.zeros(L)
    n_obs = np.zeros(L)
    q2_sum = np.zeros(L)
    n2_obs = np.zeros(L)
    any_day = np.zeros(L)
    bl = boardlike_windows(pq.panel) if pq.split == "train" else {}
    for arr, k in src:
        sp = np.asarray(arr[k, lo:hi], float).copy()
        if pq.split == "train":
            if drop_rate is not None and rng is not None:
                sp[rng.random(sp.shape) < drop_rate] = np.nan
            for T2 in bl.get(k, []):  # blank the 90-minute gap after each evaluation-like window of that day
                a, b = max(lo, T2 + 1), min(hi, T2 + 19)
                if b > a:
                    sp[a - lo : b - lo] = np.nan
        vis = np.isfinite(sp)
        q = vis & (sp <= vc[None, :])
        q_sum += q.sum(0)
        n_obs += vis.sum(0)
        any_day += q.any(0)
        s2 = slice(lo2 - lo, hi2 - lo)
        q2_sum += q[s2].sum(0)
        n2_obs += vis[s2].sum(0)
    nd = float(len(earlier))
    with np.errstate(all="ignore"):
        cols = {
            "inm_freq": np.where(n_obs > 0, q_sum / np.maximum(n_obs, 1), np.nan),
            "inm_freq2": np.where(n2_obs > 0, q2_sum / np.maximum(n2_obs, 1), np.nan),
            "inm_anyday": any_day / max(nd, 1),
            "inm_nobs": n_obs,
            "inm_ndays": np.full(L, nd),
            "inm_freq_up1": sh(np.where(n_obs > 0, q_sum / np.maximum(n_obs, 1), np.nan), 1),
            "inm_freq_dn1": sh(np.where(n_obs > 0, q_sum / np.maximum(n_obs, 1), np.nan), -1),
        }
    return np.stack(list(cols.values()), 1), list(cols)


def inmonth_horizon(pq, d: int, T: int, rng: np.random.Generator | None = None, drop_rate: float | None = None) -> np.ndarray:
    """(HOR, L) queue frequency at T+h (+-1 step) over the earlier days of the same month/split; NaN where never observed.

    The evaluation month's own recurrence at each horizon step (legal: earlier days only); the ongoing model runs without the
    train climatology, so this is its scenario-robust stand-in for clim_hh. Train windows mimic the evaluation windows as inmonth_feats does.
    """
    L = pq.p.n_links
    vc = pq.vcut
    month = pq.dates[d][:7]
    earlier = [k for k in range(d) if pq.dates[k][:7] == month] if pq.split == "train" else list(range(d))
    earlier = earlier[-16:]
    ecap = os.environ.get("QUEUE_INM_EVALCAP")  # diagnostics: evaluation windows see at most this many earlier days
    if ecap is not None and pq.split != "train":
        earlier = earlier[-int(ecap) :] if int(ecap) > 0 else []
    lo, hi = T, min(STEPS, T + HOR + 2)
    qs = np.zeros((hi - lo, L))
    ns = np.zeros((hi - lo, L))
    bl = boardlike_windows(pq.panel) if pq.split == "train" else {}
    for k in earlier:
        sp = np.asarray(pq.speed[k, lo:hi], float).copy()
        if pq.split == "train":
            if drop_rate is not None and rng is not None:
                sp[rng.random(sp.shape) < drop_rate] = np.nan
            for T2 in bl.get(k, []):
                a, b = max(lo, T2 + 1), min(hi, T2 + 19)
                if b > a:
                    sp[a - lo : b - lo] = np.nan
        vis = np.isfinite(sp)
        qs += vis & (sp <= vc[None, :])
        ns += vis
    out = np.full((HOR, L), np.nan)
    for h in range(1, HOR + 1):
        rows = slice(h - 1, min(h + 2, hi - lo))
        q = qs[rows].sum(0)
        n = ns[rows].sum(0)
        out[h - 1] = np.where(n > 0, q / np.maximum(n, 1), np.nan)
    return out


# ---------------------------------------------------------------------------
# v7 additions: remaining-capacity (demand vs downstream capacity) breakdown precursors
# ---------------------------------------------------------------------------
def remcap_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """(L, F) remaining capacity (downstream capacity - flow - on-ramp flow) / capacity over the history: levels, slope, neighbours."""
    dr = _dir(pq)
    sh = lambda a, k, fill=np.nan: shift(a, dr * k, fill)  # direction-consistent neighbours (QUEUE_DIRFIX)
    cap = pq.cap
    fl = np.asarray(pq.flow[d, T - HIST : T], float)
    on, _off = ramp_series(pq, d, T)  # (HIST, L)
    with np.errstate(all="ignore"):
        demand = fl + on  # what wants to enter the next link
        cap_dn = sh(cap, -1, np.nan)  # downstream link capacity
        rem = (np.where(np.isfinite(cap_dn), cap_dn, cap)[None, :] - demand) / cap[None, :]
        rem_last = np.nanmean(rem[-2:], 0)
        rem_m6 = np.nanmean(rem[-6:], 0)
        rem_min = np.nanmin(rem, 0)
        x = np.arange(HIST, dtype=float)
        m = np.isfinite(rem)
        cnt = m.sum(0)
        xm = np.where(m, x[:, None], 0).sum(0) / np.maximum(cnt, 1)
        ym = np.where(m, rem, 0).sum(0) / np.maximum(cnt, 1)
        slope = np.where(
            cnt >= 3,
            np.where(m, (x[:, None] - xm) * (rem - ym), 0).sum(0) / np.maximum(np.where(m, (x[:, None] - xm) ** 2, 0).sum(0), 1e-9),
            0.0,
        )
        dem_ratio = np.nanmean(demand[-3:], 0) / cap
    cols = {
        "rem_last": rem_last,
        "rem_m6": rem_m6,
        "rem_min": rem_min,
        "rem_slope": slope,
        "dem_ratio": dem_ratio,
        "rem_last_up1": sh(rem_last, 1),
        "rem_last_dn1": sh(rem_last, -1),
        "rem_min_nb": np.fmin(np.fmin(sh(rem_last, 1), sh(rem_last, -1)), rem_last),
        "steps_to_zero": np.where(slope < -1e-6, np.clip(-rem_last / np.where(slope < -1e-6, slope, -1e-6), 0, 60), 60.0),
    }
    return np.stack(list(cols.values()), 1), list(cols)


# ---------------------------------------------------------------------------
# v8 additions: in-month ONSET locations (where did corridor-level onsets start near this time on earlier days of the month)
# ---------------------------------------------------------------------------
def inmonth_onset_feats(
    pq, d: int, T: int, rng: np.random.Generator | None, drop_rate: float | None, cond: str
) -> tuple[np.ndarray, list[str]]:
    """(L, F) corridor-level onsets (a queue after 30 queue-free min) within +-60 min of the window's target time on earlier days."""
    dr = _dir(pq)
    sh = lambda a, k, fill=np.nan: shift(a, dr * k, fill)  # direction-consistent neighbours (QUEUE_DIRFIX)
    L = pq.p.n_links
    month = pq.dates[d][:7]
    earlier = [k for k in range(d) if pq.dates[k][:7] == month] if pq.split == "train" else list(range(d))
    earlier = earlier[-16:]
    c = T + HOR if cond == "onset" else T + 3
    lo, hi = max(6, c - 12), min(STEPS, c + 13)
    bl = boardlike_windows(pq.panel) if pq.split == "train" else {}
    cnt = np.zeros(L)
    near = np.zeros(L)
    anyday = 0.0
    dist = []
    for k in earlier:
        q = np.asarray(pq.obs[k, lo - 6 : hi]).copy()  # (6 + span, L)
        if pq.split == "train":
            for T2 in bl.get(k, []):  # the evaluation splits blank 90 minutes after each window
                a, b = max(lo - 6, T2 + 1), min(hi, T2 + 19)
                if b > a:
                    q[a - (lo - 6) : b - (lo - 6)] = False
        found = False
        for i in range(6, q.shape[0]):
            if q[i].any() and not q[i - 6 : i].any():  # corridor-level onset at step lo + i - 6
                t = lo + i - 6
                cnt += q[i]
                found = True
                dist.append(abs(t - c))
                if abs(t - c) <= 6:
                    near += q[i]
                break
        anyday += found
    nd = max(float(len(earlier)), 1.0)
    f = cnt / nd
    cols = {
        "inm_on_freq": f,
        "inm_on_near": near / nd,
        "inm_on_anyday": np.full(L, anyday / nd),
        "inm_on_dist": np.full(L, float(np.mean(dist)) if dist else np.nan),
        "inm_on_nb": np.maximum(sh(f, 1, 0.0), sh(f, -1, 0.0)),
    }
    return np.stack(list(cols.values()), 1), list(cols)


def corridor_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Corridor-level vehicle count and its trend (feature set v9).

    Over a corridor the jam length follows max(0, N - N0) / (k_jam - k_free) with N = sum of density x length (a public analysis of the
    generator reports R2 0.94-0.98 in March/April), and the change of N over the last 30 min predicts the change of the
    jam better than the tail's own trend. From the visible (filled) history T-12..T-1: N / N_crit at T-1, its change over 30 and 55 min,
    the queued share of the corridor length, and per link the load within 3 km on either side (both index directions) with its change.
    """
    p = pq.p
    L = p.n_links
    sp = np.asarray(pq.speed[d, T - HIST : T], float)
    fl = np.asarray(pq.flow[d, T - HIST : T], float)
    kc = np.asarray(p.param("k_crit"), float)
    ln = np.asarray(p.param("length_km"), float)
    with np.errstate(all="ignore"):
        k = fl / np.maximum(sp, 1.0)  # (HIST, L) veh/km, link total
        fin = np.isfinite(k)
        num = np.where(fin, k * ln[None, :], 0.0)
        den = np.where(fin, kc[None, :] * ln[None, :], 0.0)
        nr = num.sum(1) / np.maximum(den.sum(1), 1e-9)  # (HIST,) corridor load N / N_crit per step
        q = (sp <= pq.vcut[None, :]) & np.isfinite(sp)
        jl = (q * ln[None, :]).sum(1) / ln.sum()  # queued share of the corridor length per step
        pos = np.cumsum(ln) - ln / 2
        win_lo = np.zeros((L, L), bool)
        win_hi = np.zeros((L, L), bool)
        for i in range(L):
            dist = pos - pos[i]
            win_lo[i] = (dist <= 0) & (dist >= -3.0)
            win_hi[i] = (dist >= 0) & (dist <= 3.0)

        def local(t):
            """Per link at history step t: N / N_crit within 3 km on the lower and the higher index side (link included)."""
            a = num[t]
            b = den[t]
            return (win_lo @ a) / np.maximum(win_lo @ b, 1e-9), (win_hi @ a) / np.maximum(win_hi @ b, 1e-9)

        lo_last, hi_last = local(-1)
        lo_6, hi_6 = local(-7)
    ones = np.ones(L)
    X = np.column_stack(
        [
            ones * nr[-1],
            ones * (nr[-1] - nr[-7]),
            ones * (nr[-1] - nr[0]),
            ones * jl[-1],
            ones * (jl[-1] - jl[-7]),
            lo_last,
            hi_last,
            lo_last - lo_6,
            hi_last - hi_6,
        ]
    )
    return X.astype(float), ["cor_nr", "cor_dnr6", "cor_dnr12", "cor_jl", "cor_djl6", "loc3_lo", "loc3_hi", "loc3_lo_d6", "loc3_hi_d6"]


def corridor_feats2(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Multi-scale loads (feature set v10 = v9 + these): N/N_crit within 3-10 km and 10-25 km on either index side of each link at T-1 and
    their 30-min change (demand that arrives within 10-30 min in free flow; jam waves from downstream move ~10 km/h)."""
    p = pq.p
    sp = np.asarray(pq.speed[d, T - HIST : T], float)
    fl = np.asarray(pq.flow[d, T - HIST : T], float)
    kc = np.asarray(p.param("k_crit"), float)
    ln = np.asarray(p.param("length_km"), float)
    with np.errstate(all="ignore"):
        k = fl / np.maximum(sp, 1.0)
        fin = np.isfinite(k)
        num = np.where(fin, k * ln[None, :], 0.0)
        den = np.where(fin, kc[None, :] * ln[None, :], 0.0)
        pos = np.cumsum(ln) - ln / 2
        dist = pos[None, :] - pos[:, None]  # (L, L): dist[i, j] = pos_j - pos_i
        cols, names = [], []
        for lo, hi, tag in [(3.0, 10.0, "m"), (10.0, 25.0, "f")]:
            for side, sgn in (("lo", -1.0), ("hi", 1.0)):
                win = ((sgn * dist) >= lo) & ((sgn * dist) <= hi)
                last = (win @ num[-1]) / np.maximum(win @ den[-1], 1e-9)
                prev = (win @ num[-7]) / np.maximum(win @ den[-7], 1e-9)
                last = np.where(win.any(1), last, np.nan)
                prev = np.where(win.any(1), prev, np.nan)
                cols += [last, last - prev]
                names += [f"ld{tag}_{side}", f"ld{tag}_{side}_d6"]
    return np.column_stack(cols).astype(float), names


def shock_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Queue-tail kinematics (feature set v11 = v10 + these columns).

    A queue's upstream end moves with the shock speed w = (q_up - q_in) / (k_up - k_in) between the arriving and the queued state
    (Rankine-Hugoniot; negative = growing upstream). Queue runs from the queued flags at T-1, flows and densities averaged over the visible
    (filled) history T-3..T-1, in driving order: for a free link the
    shock speed of the nearest queue tail downstream of it, the distance to that tail and the time the tail needs to reach it; for a
    queued link the shock speed at its own queue's tail, its distance behind that tail, the time until a receding tail passes it, and
    the queue's length.
    """
    p = pq.p
    L = p.n_links
    ln = np.asarray(p.param("length_km"), float)
    with np.errstate(all="ignore"):
        v = np.nanmean(np.asarray(pq.speed[d, T - 3 : T], float), 0)
        q = np.nanmean(np.asarray(pq.flow[d, T - 3 : T], float), 0)
        k = q / np.maximum(v, 1.0)
    o = np.arange(L) if link_direction(pq.panel) > 0 else np.arange(L)[::-1]  # driving order
    qo, ko, lo = q[o], k[o], ln[o]
    s = np.cumsum(lo) - lo / 2  # link centres along the driving direction (km)
    Q = np.asarray(pq.obs[d, T - 1], bool)[o]  # queue runs at the last visible step
    out = np.full((L, 7), np.nan)
    runs = []
    j = 0
    while j < L:
        if Q[j]:
            a = j
            while j + 1 < L and Q[j + 1]:
                j += 1
            runs.append((a, j))
        j += 1
    prev_head = -1
    for a, b in runs:
        up = [i for i in (a - 2, a - 1) if i > prev_head and np.isfinite(qo[i]) and np.isfinite(ko[i])]
        inn = [i for i in range(a, min(a + 2, b + 1)) if np.isfinite(qo[i]) and np.isfinite(ko[i])]
        w = np.nan
        if up and inn:
            dq = qo[up].mean() - qo[inn].mean()
            dk = ko[up].mean() - ko[inn].mean()
            if abs(dk) > 5.0:
                w = float(np.clip(dq / dk, -40.0, 40.0))
        runlen = s[b] - s[a] + (lo[a] + lo[b]) / 2
        for i in range(prev_head + 1, a):  # free links between the previous head and this tail
            dist = s[a] - s[i]
            out[i, 0:3] = [w, dist, dist / -w * 60.0 if (np.isfinite(w) and w < -0.5) else 999.0]
        for i in range(a, b + 1):
            dist = s[i] - s[a]
            out[i, 3:7] = [w, dist, dist / w * 60.0 if (np.isfinite(w) and w > 0.5) else 999.0, runlen]
        prev_head = b
    X = np.empty_like(out)
    X[o] = out  # back to cache order
    return np.minimum(X, 999.0), ["sh_w_up", "sh_dist_up", "sh_eta_up", "sh_w_in", "sh_dist_in", "sh_eta_in", "sh_runlen"]


def corridor_feats3(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Distance-banded context (feature set v12 = v10 + these columns): within 0-3, 3-10 and 10-25 km on either index side of each link
    (the link itself excluded), the length-weighted mean flow / capacity and speed / v_cut at T-1 with their 30-min changes, and the
    queued share of the band at T-1 (visible, filled history)."""
    p = pq.p
    ln = np.asarray(p.param("length_km"), float)
    with np.errstate(all="ignore"):
        fr = np.asarray(pq.flow[d, T - HIST : T], float) / pq.cap[None, :]
        vr = np.asarray(pq.speed[d, T - HIST : T], float) / pq.vcut[None, :]
        q = np.asarray(pq.obs[d, T - HIST : T], float)
        pos = np.cumsum(ln) - ln / 2
        dist = pos[None, :] - pos[:, None]  # (L, L): dist[i, j] = pos_j - pos_i
        cols, names = [], []
        for lo, hi, tag in [(0.0, 3.0, "n"), (3.0, 10.0, "m"), (10.0, 25.0, "f")]:
            for side, sgn in (("lo", -1.0), ("hi", 1.0)):
                wl = (((sgn * dist) > lo) & ((sgn * dist) <= hi)) * ln[None, :]

                def band(a, wl=wl):
                    """(L,) length-weighted mean of a over each link's band (finite entries only; NaN where none)."""
                    f = np.isfinite(a)
                    return (wl @ np.where(f, a, 0.0)) / (wl @ f.astype(float))

                fr1, fr7, vr1, vr7 = band(fr[-1]), band(fr[-7]), band(vr[-1]), band(vr[-7])
                cols += [fr1, fr1 - fr7, vr1, vr1 - vr7, band(q[-1])]
                names += [f"bd{tag}_{side}_fr", f"bd{tag}_{side}_fr_d6", f"bd{tag}_{side}_vr", f"bd{tag}_{side}_vr_d6", f"bd{tag}_{side}_q"]
    return np.column_stack(cols).astype(float), names


def edge_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Queue-edge trends and a raw speed patch (feature set v13 = v10 + these columns).

    The jam's head stays put and its tail creeps (by a public analysis of the generator, the tail's own 30-min trend beats the
    shock formula), so per link, in driving order: the distance (km) to the nearest queued link downstream / upstream at T-1 and its
    change since T-4 and T-7 (visible, filled queued flags); and speed / v_cut of the 1..8 links downstream and upstream at T-1 and T-6.
    """
    p = pq.p
    L = p.n_links
    ln = np.asarray(p.param("length_km"), float)
    dr = link_direction(pq.panel)
    o = np.arange(L) if dr > 0 else np.arange(L)[::-1]  # driving order
    s = np.cumsum(ln[o]) - ln[o] / 2

    def dists(t):
        """Per link in driving order: km to the nearest queued link at or downstream and at or upstream at step t (capped at 30)."""
        q = np.asarray(pq.obs[d, t], bool)[o]
        qi = np.where(q)[0]
        dn = np.full(L, 30.0)
        up = np.full(L, 30.0)
        if len(qi):
            j = np.searchsorted(qi, np.arange(L))  # first queued at or downstream
            ok = j < len(qi)
            dn[ok] = s[qi[j[ok]]] - s[ok]
            j2 = np.searchsorted(qi, np.arange(L), side="right") - 1  # last queued at or upstream
            ok2 = j2 >= 0
            up[ok2] = s[ok2] - s[qi[j2[ok2]]]
        return np.minimum(dn, 30.0), np.minimum(up, 30.0)

    dn1, up1 = dists(T - 1)
    dn4, up4 = dists(T - 4)
    dn7, up7 = dists(T - 7)
    cols = [dn1, dn1 - dn4, dn1 - dn7, up1, up1 - up4, up1 - up7]
    names = ["eg_dn", "eg_dn_d3", "eg_dn_d6", "eg_up", "eg_up_d3", "eg_up_d6"]
    with np.errstate(all="ignore"):
        for t, tag in ((T - 1, "1"), (T - 6, "6")):
            vr = (np.asarray(pq.speed[d, t], float) / pq.vcut)[o]
            for k in range(1, 9):
                dn = np.full(L, np.nan)
                dn[:-k] = vr[k:]
                up = np.full(L, np.nan)
                up[k:] = vr[:-k]
                cols += [dn, up]
                names += [f"pv{tag}_dn{k}", f"pv{tag}_up{k}"]
    X = np.column_stack(cols)
    out = np.empty_like(X)
    out[o] = X  # back to cache order
    return out.astype(float), names


def inmonth_day_feats(pq, d: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Whole-day in-month site activity (feature set v16 = v10 + these columns).

    inmonth_feats looks at the window's own time of day (+-30 min) on the month's earlier days; with the 0-3 earlier days of a scored
    window that misses sites that were active at other hours. Per link over the same earlier days (06:00-20:00): the share of steps
    queued, the share of days with any queued step, and the neighbourhood maxima (+-2 links) of both.
    """
    L = pq.p.n_links
    vc = pq.vcut
    month = pq.dates[d][:7]
    earlier = [k for k in range(d) if pq.dates[k][:7] == month] if pq.split == "train" else list(range(d))
    earlier = earlier[-16:]
    ecap = os.environ.get("QUEUE_INM_EVALCAP")
    if ecap is not None and pq.split != "train":
        earlier = earlier[-int(ecap) :] if int(ecap) > 0 else []
    q_sum = np.zeros(L)
    n_obs = np.zeros(L)
    any_day = np.zeros(L)
    for k in earlier:
        sp = np.asarray(pq.speed[k, 72:240], float)
        vis = np.isfinite(sp)
        q = vis & (sp <= vc[None, :])
        q_sum += q.sum(0)
        n_obs += vis.sum(0)
        any_day += q.any(0)
    nd = float(len(earlier))
    with np.errstate(all="ignore"):
        share = np.where(n_obs > 0, q_sum / np.maximum(n_obs, 1), np.nan)
        anyd = any_day / nd if nd > 0 else np.full(L, np.nan)

    def nbmax(a):
        """(L,) maximum of a over each link and the two links on either side (NaN-ignoring)."""
        out = np.full(L, np.nan)
        for kk in (-2, -1, 0, 1, 2):
            out = np.fmax(out, shift(a, kk))
        return out

    return np.column_stack([share, anyd, nbmax(share), nbmax(anyd)]).astype(float), [
        "inmd_share",
        "inmd_anyday",
        "inmd_share_nb",
        "inmd_anyday_nb",
    ]


def inmonth_demand_feats(pq, d: int, T: int, cond: str) -> tuple[np.ndarray, list[str]]:
    """In-month demand (feature set v17 = v16 + these columns): per link over the month's earlier days, the mean flow / capacity and density /
    k_crit within +-30 min of the window's target time (T+30 for onset, T+15 for ongoing) and over the whole day 06:00-20:00 (95th percentile
    of flow / capacity), plus +-2-link maxima. Queue recurrence needs a site to have queued already; its load is visible every day."""
    p = pq.p
    L = p.n_links
    month = pq.dates[d][:7]
    earlier = [k for k in range(d) if pq.dates[k][:7] == month] if pq.split == "train" else list(range(d))
    earlier = earlier[-16:]
    ecap = os.environ.get("QUEUE_INM_EVALCAP")
    if ecap is not None and pq.split != "train":
        earlier = earlier[-int(ecap) :] if int(ecap) > 0 else []
    names = ["inmq_fr", "inmq_kr", "inmq_fr95", "inmq_fr_nb", "inmq_kr_nb", "inmq_fr95_nb"]
    if not earlier:
        return np.full((L, 6), np.nan), names
    c = T + HOR if cond == "onset" else T + 3
    lo, hi = max(0, c - 6), min(STEPS, c + 7)
    kc = np.asarray(p.param("k_crit"), float)
    with np.errstate(all="ignore"):
        fl = np.stack([np.asarray(pq.flow[k, lo:hi], float) for k in earlier])
        sp = np.stack([np.asarray(pq.speed[k, lo:hi], float) for k in earlier])
        fr = np.nanmean((fl / pq.cap[None, None, :]).reshape(-1, L), 0)
        kr = np.nanmean((fl / np.maximum(sp, 1.0) / kc[None, None, :]).reshape(-1, L), 0)
        fd = np.stack([np.asarray(pq.flow[k, 72:240], float) for k in earlier]) / pq.cap[None, None, :]
        fr95 = np.nanpercentile(fd.reshape(-1, L), 95, axis=0)

    def nbmax(a):
        """(L,) maximum of a over each link and the two links on either side (NaN-ignoring)."""
        out = np.full(L, np.nan)
        for kk in (-2, -1, 0, 1, 2):
            out = np.fmax(out, shift(a, kk))
        return out

    return np.column_stack([fr, kr, fr95, nbmax(fr), nbmax(kr), nbmax(fr95)]).astype(float), names
