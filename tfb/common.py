"""Shared paths, panel lists and the array cache loader.

The cache holds one directory per panel with ``.npy`` arrays shaped
``(days, 288, links)`` for the mainline channels and ``(days, 288, ramps)`` for
the ramps, plus ``meta.json`` with the link order and static parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RELEASE = PROJECT / "data" / "kaggle_public"
CACHE = PROJECT / "cache_top3"

PANELS = [
    "D7_I10_E",
    "D7_I10_W",
    "D7_I210_E",
    "D7_I210_W",
    "D7_I405_N",
    "D7_I405_S",
    "D12_I5_N",
    "D12_I5_S",
    "D12_I405_N",
    "D12_I405_S",
]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
QUEUE_PANELS = [p for p in PANELS if FAMILY[p] != "D12_I405"]
SPLITS = ["train", "validation", "private"]
STEPS = 288
REGIMES = ("R1", "R2", "R3")

# score weights from config/competition_contract.json
W_STATE, W_QUEUE, W_PHYS, W_ODME = 0.35, 0.30, 0.15, 0.20
SPEED_W, FLOW_W, SPEED_NORM, FLOW_NORM = 0.54, 0.46, 25.0, 600.0


def t_index(ts: np.ndarray | list[str]) -> np.ndarray:
    """Slot index 0..287 from ISO timestamps 'YYYY-MM-DDTHH:MM:SSZ'."""
    s = np.asarray(ts, dtype=str)
    hh = np.char.slice(s, 11, 13).astype(int)
    mm = np.char.slice(s, 14, 16).astype(int)
    return hh * 12 + mm // 5


@dataclass
class Panel:
    """One panel's cached arrays for one split (lazy, memory-mapped)."""

    name: str
    split: str
    meta: dict
    _arrays: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        """Cache directory of the panel (cache_top3/<panel>)."""
        return CACHE / self.name

    @property
    def links(self) -> list[str]:
        """Link ids in cache order (the link axis of the arrays)."""
        return self.meta["links"]

    @property
    def n_links(self) -> int:
        """Number of links."""
        return len(self.meta["links"])

    @property
    def dates(self) -> list[str]:
        """The split's dates ('YYYY-MM-DD', sorted) in the day-axis order of the arrays."""
        return self.meta["dates"][self.split]

    @property
    def regimes(self) -> np.ndarray:
        """Mask regime (R1, R2 or R3) of each day of the split."""
        return np.asarray(self.meta["regimes"][self.split])

    def param(self, key: str) -> np.ndarray:
        """Per-link static parameter `key` from meta.json (e.g. v_cut, capacity) as a float64 array (L,)."""
        return np.asarray(self.meta["params"][key], dtype=np.float64)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            self._arrays[key] = np.load(self.dir / f"{self.split}_{key}.npy", mmap_mode="r")
        return self._arrays[key]

    def has(self, key: str) -> bool:
        """Whether the cache holds <split>_<key>.npy for this panel."""
        return (self.dir / f"{self.split}_{key}.npy").exists()


def load_panel(name: str, split: str = "train") -> Panel:
    """Panel of one split from cache_top3/<name>/meta.json; the arrays load lazily on first access."""
    meta = json.loads((CACHE / name / "meta.json").read_text())
    return Panel(name, split, meta)
