"""Data pipeline for the Task-1 inpainting model.

All panels are held in RAM as numpy arrays; batches are assembled on the
training device (fresh random masks, linear-interpolation baseline, feature
channels) so that the CPU never becomes the bottleneck.

Channel layout of the model input (see ``FEATURES``): observed values (zero
where blank), an observed-mask channel, pct, the linear-interpolation baseline
of the three channels, time-of-day and weekday encodings, per-link statics and
the attached on-/off-ramp flows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from tfb.common import PANELS, PROJECT, load_panel

SPEED_SCALE, FLOW_SCALE, RAMP_SCALE = 100.0, 2000.0, 1000.0
# input normalisation: centred on the link's free speed / typical flow, scaled to the residual magnitude
V_NORM, QL_CENTRE, QL_NORM, OCC_CENTRE, OCC_NORM = 20.0, 1000.0, 500.0, 0.10, 0.10
SPEED_MAX = 120.75
FEATURES = [
    "v_obs",
    "ql_obs",
    "occ_obs",
    "obs",
    "pct",
    "v_lin",
    "ql_lin",
    "occ_lin",
    "gap",
    "tod_sin",
    "tod_cos",
    "wd_sin",
    "wd_cos",
    "weekend",
    "lanes",
    "vf",
    "cap_lane",
    "length",
    "kjam_lane",
    "on_att",
    "off_att",
    "ramp_on",
    "ramp_off",
    "ramp_on_ok",
    "ramp_off_ok",
]
N_FEATURES = len(FEATURES)


def holdout_dates(name: str = "holdout31") -> list[str]:
    """Dates of a named holdout set from config/<name>.json (default holdout31)."""
    return json.loads((PROJECT / "config" / f"{name}.json").read_text())["holdout_dates"]


def february_dates() -> list[str]:
    """The train split's February 2031 dates (read from the first panel)."""
    return [d for d in load_panel(PANELS[0], "train").dates if d.startswith("2031-02")]


def ramp_link_flows(p) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-link attached ramp flow (days, T, L) for on- and off-ramps plus validity flags."""
    rf = np.asarray(p["ramp_flow"], dtype=np.float32)
    re = np.asarray(p["ramp_elig"])
    valid = re & np.isfinite(rf)
    rl = np.asarray(p.meta["ramps"]["ramp_link"])
    rt = np.asarray(p.meta["ramps"]["ramp_type"])
    D, T, _ = rf.shape
    L = p.n_links
    on = np.zeros((D, T, L), np.float32)
    off = np.zeros((D, T, L), np.float32)
    on_ok = np.zeros((D, T, L), bool)
    off_ok = np.zeros((D, T, L), bool)
    for i, (typ, l) in enumerate(zip(rt, rl)):
        if l < 0:
            continue
        f = np.where(valid[:, :, i], rf[:, :, i], 0.0)
        if typ.upper() in ("OR", "ON"):
            on[:, :, l] += f
            on_ok[:, :, l] |= valid[:, :, i]
        else:
            off[:, :, l] += f
            off_ok[:, :, l] |= valid[:, :, i]
    return on, off, on_ok, off_ok


def transition_ok(p, days: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """(D, T-1, L) bool: transitions counted by the physics instrument (within a day)."""
    from tfb.scoring import _ramp_validity

    on_valid, off_valid = _ramp_validity(p, days)
    return elig[:, :-1] & elig[:, 1:] & on_valid[:, :-1] & off_valid[:, :-1]


@dataclass
class PanelArrays:
    """One panel + split held in RAM."""

    name: str
    split: str
    dates: list[str]
    regimes: np.ndarray
    lanes: np.ndarray
    length: np.ndarray
    static: np.ndarray  # (L, n_static) float32
    fallback_v: np.ndarray  # (L,) value used when a link has no observation all day
    fallback_ql: np.ndarray
    fallback_occ: np.ndarray
    v: np.ndarray  # (D, T, L) truth (train) or masked layer (other splits); NaN = blank
    q: np.ndarray
    occ: np.ndarray
    pct: np.ndarray  # (D, T, L) int8
    elig: np.ndarray  # bool
    target: np.ndarray  # bool
    ramp_on: np.ndarray
    ramp_off: np.ndarray
    ramp_on_ok: np.ndarray
    ramp_off_ok: np.ndarray
    weekday: np.ndarray  # (D,) int
    trans_ok: np.ndarray | None = None  # (D, T-1, L) bool, train split only
    panel: object = field(default=None, repr=False)

    @property
    def n_links(self) -> int:
        """Number of links (last axis of v)."""
        return self.v.shape[2]

    @property
    def n_days(self) -> int:
        """Number of days (first axis of v)."""
        return self.v.shape[0]


def load_arrays(name: str, split: str, with_trans: bool = True) -> PanelArrays:
    """One panel and split in RAM: train truth or masked layer, scaled statics, ramp flows, fallbacks; trans_ok if train and with_trans."""
    p = load_panel(name, split)
    truth = split == "train"
    v = np.asarray(p["speed" if truth else "speed_m"], dtype=np.float32)
    q = np.asarray(p["flow" if truth else "flow_m"], dtype=np.float32)
    occ = np.asarray(p["occ" if truth else "occ_m"], dtype=np.float32)
    pct = np.asarray(p["pct"])
    elig = np.asarray(p["elig"])
    target = np.asarray(p["target"])
    lanes = p.param("lanes")
    length = p.param("length_km")
    vf = p.param("free_speed")
    cap = p.param("capacity")
    kj = p.param("k_jam")
    on_att = np.asarray(p.meta["params"]["on_attached"], dtype=np.float32)
    off_att = np.asarray(p.meta["params"]["off_attached"], dtype=np.float32)
    static = np.stack(
        [lanes / 6.0, vf / 120.0, cap / lanes / 2500.0, np.minimum(length, 4.0) / 2.0, kj / lanes / 150.0, on_att, off_att], axis=1
    ).astype(np.float32)
    ron, roff, ron_ok, roff_ok = ramp_link_flows(p)
    weekday = np.array([pd.Timestamp(d).weekday() for d in p.dates])
    fb_v = vf.astype(np.float32)
    fb_ql = (0.3 * cap / lanes).astype(np.float32)
    fb_occ = np.full(p.n_links, 0.08, np.float32)
    arr = PanelArrays(
        name,
        split,
        list(p.dates),
        p.regimes,
        lanes.astype(np.float32),
        length.astype(np.float32),
        static,
        fb_v,
        fb_ql,
        fb_occ,
        v,
        q,
        occ,
        pct,
        elig,
        target,
        ron,
        roff,
        ron_ok,
        roff_ok,
        weekday,
        panel=p,
    )
    if truth and with_trans:
        arr.trans_ok = transition_ok(p, np.arange(len(p.dates)), elig)
    return arr


# ----------------------------------------------------------------------------- torch helpers
def time_features(T: int, weekday: torch.Tensor, device) -> torch.Tensor:
    """(B, 5, T) time-of-day sin/cos, weekday sin/cos, weekend flag."""
    t = torch.arange(T, device=device, dtype=torch.float32) / T * 2 * np.pi
    B = weekday.shape[0]
    tod = torch.stack([torch.sin(t), torch.cos(t)]).unsqueeze(0).expand(B, 2, T)
    wd = weekday.to(device).float() / 7 * 2 * np.pi
    wds = torch.stack([torch.sin(wd), torch.cos(wd), (weekday.to(device) >= 5).float()], dim=1)  # (B,3)
    return torch.cat([tod, wds.unsqueeze(-1).expand(B, 3, T)], dim=1)


class TrainSampler:
    """Draws (panel, day, link-crop) batches with fresh synthetic masks on the device."""

    def __init__(
        self,
        panels: list[PanelArrays],
        train_days: dict[str, np.ndarray],
        crop: int,
        batch: int,
        device,
        rates=(0.2, 0.3, 0.5),
        rate_jitter: float = 0.1,
        p_band: float = 0.2,
        band_len: int = 18,
        seed: int = 42,
        forecast: bool = False,
        band_windows: dict | None = None,
        p_band_win: float = 0.8,
        band_loss: bool = False,
    ):
        self.forecast = forecast
        # blackout mode: bands start right after a Task-2 window origin T (T+1..T+18, like the evaluation months' blackouts),
        # the published raw history T-12..T stays observed, and with band_loss every eligible band cell is a loss cell
        self.band_windows = band_windows or {}
        self.p_band_win = p_band_win
        self.band_loss = band_loss
        self.panels = panels
        self.train_days = train_days
        self.crop = crop
        self.batch = batch
        self.device = device
        self.rates = rates
        self.rate_jitter = rate_jitter
        self.p_band = p_band
        self.band_len = band_len
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        w = np.array([max(a.n_links, crop) * len(train_days[a.name]) for a in panels], dtype=float)
        self.pw = w / w.sum()
        self._static_cache = {}

    def _static(self, a: PanelArrays):
        if a.name not in self._static_cache:
            d = self.device
            self._static_cache[a.name] = {
                "static": torch.from_numpy(a.static).to(d),
                "lanes": torch.from_numpy(a.lanes).to(d),
                "length": torch.from_numpy(a.length).to(d),
                "fb_v": torch.from_numpy(a.fallback_v).to(d),
                "fb_ql": torch.from_numpy(a.fallback_ql).to(d),
                "fb_occ": torch.from_numpy(a.fallback_occ).to(d),
            }
        return self._static_cache[a.name]

    def sample(self) -> dict:
        """One batch: B days of one panel (weights pw), a W-link crop and a fresh synthetic mask, as a dict of (B, T, W) device tensors.

        obs: visible cells; loss_mask: blanked eligible cells (with band_loss also band cells; in forecast mode the observed T+1..T+6).
        """
        pi = self.rng.choice(len(self.panels), p=self.pw)
        a = self.panels[pi]
        L = a.n_links
        W = self.crop
        days = self.rng.choice(self.train_days[a.name], size=self.batch, replace=len(self.train_days[a.name]) < self.batch)
        if L > W:
            starts = self.rng.integers(0, L - W + 1, size=self.batch)
        else:
            starts = np.zeros(self.batch, dtype=int)
        sl = lambda arr, pad_val: self._crop(arr, days, starts, W, pad_val)
        d = self.device
        v = torch.from_numpy(sl(a.v, np.nan)).to(d)
        q = torch.from_numpy(sl(a.q, np.nan)).to(d)
        occ = torch.from_numpy(sl(a.occ, np.nan)).to(d)
        pct = torch.from_numpy(sl(a.pct, 0)).to(d)
        elig = torch.from_numpy(sl(a.elig, False)).to(d)
        ron = torch.from_numpy(sl(a.ramp_on, 0.0)).to(d)
        roff = torch.from_numpy(sl(a.ramp_off, 0.0)).to(d)
        ron_ok = torch.from_numpy(sl(a.ramp_on_ok, False)).to(d)
        roff_ok = torch.from_numpy(sl(a.ramp_off_ok, False)).to(d)
        trans = torch.from_numpy(sl(a.trans_ok, False)).to(d) if a.trans_ok is not None else None
        st = self._static(a)
        # statics for the crop (pad with the first link's values, masked out of the loss anyway)
        lidx = np.clip(starts[:, None] + np.arange(W)[None, :], 0, L - 1)  # (B, W)
        lidx_t = torch.from_numpy(lidx).to(d)
        static = st["static"][lidx_t]  # (B, W, n_static)
        lanes = st["lanes"][lidx_t]
        length = st["length"][lidx_t]
        fb_v = st["fb_v"][lidx_t]
        fb_ql = st["fb_ql"][lidx_t]
        fb_occ = st["fb_occ"][lidx_t]
        finite = torch.isfinite(v)
        # synthetic mask: per-sample rate, drawn over eligible cells; optional 90-minute bands
        B, T, _ = v.shape
        r = torch.tensor(self.rng.choice(self.rates, size=B) * (1 - self.rate_jitter * self.rng.random(B)), dtype=torch.float32, device=d)
        u = torch.rand((B, T, W), generator=self.gen, device=d)
        blank = elig & (u < r.view(B, 1, 1))
        band = torch.zeros((B, T, W), dtype=torch.bool, device=d)
        hist_rows = torch.zeros((B, T, 1), dtype=torch.bool, device=d)
        wins_p = self.band_windows.get(a.name, {})
        for b in range(B):
            if self.rng.random() < self.p_band:
                wins = wins_p.get(int(days[b]), [])
                for _ in range(self.rng.integers(1, 3)):
                    if wins and self.rng.random() < self.p_band_win:
                        T0 = int(self.rng.choice(wins))
                        s = T0 + 1
                        hist_rows[b, max(0, T0 - 12) : T0 + 1] = True
                    else:
                        s = int(self.rng.integers(72, T - 36 - self.band_len))
                    band[b, s : s + self.band_len, :] = True
        blank = blank & ~hist_rows  # the published history is raw
        obs = finite & ~blank & ~band
        loss_mask = blank  # eligible & synthetic target
        if self.band_loss:
            loss_mask = loss_mask | (band & elig & finite)
        if self.forecast:  # forecasting: everything after a random origin is hidden
            Tb = torch.from_numpy(self.rng.integers(24, T - 7, size=B)).to(d).view(B, 1, 1)
            tt = torch.arange(T, device=d).view(1, T, 1)
            hist = (tt >= Tb - 12) & (tt < Tb)  # the published raw history window
            future = tt > Tb
            blank_f = blank & ~hist  # Task-1-like mask before T-12 and at T
            obs = finite & ~blank_f & ~future
            loss_mask = finite & (tt > Tb) & (tt <= Tb + 6)  # T+5 .. T+30, all observed cells
            ron = ron.masked_fill(future.expand_as(ron), 0.0)
            roff = roff.masked_fill(future.expand_as(roff), 0.0)
            ron_ok = ron_ok & ~future
            roff_ok = roff_ok & ~future
            pct = pct.masked_fill(future.expand_as(pct), 0)
            elig = elig & ~future
        weekday = torch.from_numpy(a.weekday[days])
        return {
            "v": v,
            "q": q,
            "occ": occ,
            "obs": obs,
            "pct": pct,
            "elig": elig,
            "loss_mask": loss_mask,
            "trans": trans,
            "ramp_on": ron,
            "ramp_off": roff,
            "ramp_on_ok": ron_ok,
            "ramp_off_ok": roff_ok,
            "weekday": weekday,
            "static": static,
            "lanes": lanes,
            "length": length,
            "fb_v": fb_v,
            "fb_ql": fb_ql,
            "fb_occ": fb_occ,
            "panel": a.name,
            "days": days,
            "starts": starts,
        }

    @staticmethod
    def _crop(arr: np.ndarray, days: np.ndarray, starts: np.ndarray, W: int, pad_val) -> np.ndarray:
        L = arr.shape[2]
        out = np.empty((len(days),) + arr.shape[1:2] + (W,), dtype=arr.dtype)
        if L >= W:
            for i, (dd, s) in enumerate(zip(days, starts)):
                out[i] = arr[dd, :, s : s + W]
        else:
            out[...] = pad_val
            for i, dd in enumerate(days):
                out[i, :, :L] = arr[dd]
        return out


def batch_inputs(sample: dict):
    """Feature tensor (B, C, T, W) for a sampler batch plus the interpolation baselines (physical units)."""
    v, q, occ, obs = sample["v"], sample["q"], sample["occ"], sample["obs"]
    B, T, W = v.shape
    lanes = sample["lanes"]  # (B, W)
    vf = sample["fb_v"].view(B, 1, W)
    vz = torch.where(obs, v, vf.expand(B, T, W))
    ql = q / lanes.view(B, 1, W)
    qlz = torch.where(obs, ql, torch.full_like(ql, QL_CENTRE))
    oz = torch.where(obs, occ, torch.full_like(occ, OCC_CENTRE))
    v_lin, gap = interp_time(vz, obs, sample["fb_v"], return_gap=True)
    ql_lin = interp_time(qlz, obs, sample["fb_ql"])
    o_lin = interp_time(oz, obs, sample["fb_occ"])
    tf = time_features(T, sample["weekday"], v.device)
    obsf = obs.float()
    chans = [
        (vz - vf) / V_NORM * obsf,
        (qlz - QL_CENTRE) / QL_NORM * obsf,
        (oz - OCC_CENTRE) / OCC_NORM * obsf,
        obsf,
        sample["pct"].float() / 100.0,
        (v_lin - vf) / V_NORM,
        (ql_lin - QL_CENTRE) / QL_NORM,
        (o_lin - OCC_CENTRE) / OCC_NORM,
        (gap.float() / 12.0).clamp(max=4.0),
    ]
    x = torch.stack(chans, dim=1)
    tfe = tf.unsqueeze(-1).expand(B, 5, T, W)
    st = sample["static"].permute(0, 2, 1).unsqueeze(2).expand(B, sample["static"].shape[2], T, W)
    l3 = lanes.view(B, 1, W)
    ramps = torch.stack(
        [
            sample["ramp_on"] / l3 / RAMP_SCALE,
            sample["ramp_off"] / l3 / RAMP_SCALE,
            sample["ramp_on_ok"].float(),
            sample["ramp_off_ok"].float(),
        ],
        dim=1,
    )
    x = torch.cat([x, tfe, st, ramps], dim=1)
    return x, v_lin, ql_lin


def interp_time(x: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor, return_gap: bool = False):
    """Linear interpolation along dim 1 of x (B, T, L) over observed cells ``mask``.

    Edges are extended with the nearest observation; links without any
    observation get the per-sample ``fallback`` (B, L) broadcast over time.
    """
    B, T, L = x.shape
    idx = torch.arange(T, device=x.device).view(1, T, 1).expand(B, T, L)
    neg = torch.full_like(idx, -1)
    prev = torch.where(mask, idx, neg).cummax(dim=1).values
    nxt = T - 1 - torch.where(mask.flip(1), idx, neg).cummax(dim=1).values.flip(1)
    has_prev = prev >= 0
    has_next = nxt <= T - 1
    xp = torch.gather(x, 1, prev.clamp(min=0))
    xn = torch.gather(x, 1, nxt.clamp(max=T - 1))
    denom = (nxt - prev).clamp(min=1).to(x.dtype)
    w = (idx - prev).to(x.dtype) / denom
    out = torch.where(has_prev & has_next, xp + w * (xn - xp), torch.where(has_prev, xp, xn))
    out = torch.where(~(has_prev | has_next), fallback.view(B, 1, L).expand(B, T, L).to(x.dtype), out)
    if return_gap:
        gap = torch.where(mask, torch.zeros_like(idx), (nxt.clamp(max=T) - prev.clamp(min=-1)).clamp(max=T))
        return out, gap
    return out


def full_panel_batch(a: PanelArrays, days: np.ndarray, device, pad_to: int = 8) -> dict:
    """Whole-panel inputs for inference on the given days with the REAL masks (blank = NaN)."""
    L = a.n_links
    W = int(np.ceil(L / pad_to) * pad_to)
    starts = np.zeros(len(days), dtype=int)
    sl = lambda arr, pad_val: TrainSampler._crop(arr, days, starts, W, pad_val) if W > L else arr[days]
    d = device
    v = torch.from_numpy(np.ascontiguousarray(sl(a.v, np.nan))).to(d)
    q = torch.from_numpy(np.ascontiguousarray(sl(a.q, np.nan))).to(d)
    occ = torch.from_numpy(np.ascontiguousarray(sl(a.occ, np.nan))).to(d)
    pct = torch.from_numpy(np.ascontiguousarray(sl(a.pct, 0))).to(d)
    elig = torch.from_numpy(np.ascontiguousarray(sl(a.elig, False))).to(d)
    ron = torch.from_numpy(np.ascontiguousarray(sl(a.ramp_on, 0.0))).to(d)
    roff = torch.from_numpy(np.ascontiguousarray(sl(a.ramp_off, 0.0))).to(d)
    ron_ok = torch.from_numpy(np.ascontiguousarray(sl(a.ramp_on_ok, False))).to(d)
    roff_ok = torch.from_numpy(np.ascontiguousarray(sl(a.ramp_off_ok, False))).to(d)
    B = len(days)
    lidx = np.clip(np.arange(W), 0, L - 1)
    static = torch.from_numpy(a.static[lidx]).to(d).unsqueeze(0).expand(B, W, a.static.shape[1])
    lanes = torch.from_numpy(a.lanes[lidx]).to(d).unsqueeze(0).expand(B, W)
    length = torch.from_numpy(a.length[lidx]).to(d).unsqueeze(0).expand(B, W)
    fb_v = torch.from_numpy(a.fallback_v[lidx]).to(d).unsqueeze(0).expand(B, W)
    fb_ql = torch.from_numpy(a.fallback_ql[lidx]).to(d).unsqueeze(0).expand(B, W)
    fb_occ = torch.from_numpy(a.fallback_occ[lidx]).to(d).unsqueeze(0).expand(B, W)
    target = torch.from_numpy(np.ascontiguousarray(sl(a.target, False))).to(d)
    obs = torch.isfinite(v) & ~target  # train split: truth layer, hide the real targets
    return {
        "v": v,
        "q": q,
        "occ": occ,
        "obs": obs,
        "pct": pct,
        "elig": elig,
        "target": target,
        "ramp_on": ron,
        "ramp_off": roff,
        "ramp_on_ok": ron_ok,
        "ramp_off_ok": roff_ok,
        "weekday": torch.from_numpy(a.weekday[days]),
        "static": static,
        "lanes": lanes,
        "length": length,
        "fb_v": fb_v,
        "fb_ql": fb_ql,
        "fb_occ": fb_occ,
        "n_links": L,
    }
