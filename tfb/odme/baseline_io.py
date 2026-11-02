"""Readers for the Task-4 network and released counts/prior (same logic as the organizers' public baseline,
src/task4/build_task4_odme_artifacts.py in github.com/jacky850/trafficflowbench-public, MIT licence), so the ODME solve needs no
checkout of that repository."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def released_counts(release: Path, panel: str, split: str) -> pd.DataFrame | None:
    """The link counts the release publishes for this split (task4/<panel>/<split>/synthetic_link_counts.csv)."""
    path = release / "task4" / panel / split / "synthetic_link_counts.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"link_id": str})
    return frame[["link_id", "count"]]


def released_prior(release: Path, panel: str, split: str) -> pd.DataFrame | None:
    """The weak path-flow prior the release publishes for this split."""
    path = release / "task4" / panel / split / "synthetic_weak_prior.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"path_id": str})


def load_operator(network: Path) -> tuple[list[str], list[str], np.ndarray, pd.DataFrame]:
    """Path ids, link ids, the 0/1 path-link incidence operator (links x paths) and the path table."""
    paths = pd.read_csv(network / "path_set.csv")
    paths["path_id"] = paths.path_id.astype(str)
    incidence = pd.read_csv(network / "path_link_incidence.csv")
    incidence["path_id"] = incidence.path_id.astype(str)
    incidence["link_id"] = incidence.link_id.astype(str)
    path_ids = paths.path_id.tolist()
    link_ids = incidence.link_id.drop_duplicates().tolist()
    path_index = {x: i for i, x in enumerate(path_ids)}
    link_index = {x: i for i, x in enumerate(link_ids)}
    rr = incidence.link_id.map(link_index).to_numpy(dtype=np.int64)
    cc = incidence.path_id.map(path_index).to_numpy(dtype=np.int64)
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    A[rr, cc] = 1.0
    return path_ids, link_ids, A, paths


def prior_values(release: Path, panel: str, split: str, path_ids: list[str]) -> np.ndarray:
    """The split's own released weak prior, in path order, zero where absent."""
    frame = released_prior(release, panel, split)
    if frame is None:
        return np.zeros(len(path_ids), dtype=float)
    frame = frame.set_index(frame.path_id.astype(str)).reindex(path_ids)
    return pd.to_numeric(frame.path_flow, errors="coerce").fillna(0.0).to_numpy(dtype=float)
