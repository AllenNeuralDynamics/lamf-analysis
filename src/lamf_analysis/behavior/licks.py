"""Lick behavior metrics."""

import numpy as np


def get_lick_bout_start_times(
    lick_times,
    inter_lick_interval_threshold: float = 0.7,
) -> np.ndarray:
    """Return the first lick time in each bout.

    Consecutive licks separated by no more than
    ``inter_lick_interval_threshold`` seconds belong to the same bout.
    """
    lick_times = np.asarray(lick_times, dtype=float)
    if lick_times.ndim != 1:
        raise ValueError("lick_times must be one-dimensional")
    if inter_lick_interval_threshold <= 0:
        raise ValueError("inter_lick_interval_threshold must be positive")
    if not np.all(np.isfinite(lick_times)):
        raise ValueError("lick_times must contain only finite values")
    if lick_times.size == 0:
        return lick_times.copy()

    lick_times = np.sort(lick_times)
    bout_start = np.concatenate(
        ([True], np.diff(lick_times) > inter_lick_interval_threshold)
    )
    return lick_times[bout_start]


def add_lick_bouts(bsd, inter_lick_interval_threshold: float = 0.7):
    """Annotate a BehaviorSessionDataset lick table with bout boundaries."""
    licks = bsd.licks.data
    licks["pre_ILI"] = (
        licks["timestamps"] - licks["timestamps"].shift(fill_value=-10)
    )
    licks["post_ILI"] = (
        licks["timestamps"].shift(periods=-1, fill_value=5000)
        - licks["timestamps"]
    )
    licks["bout_start"] = licks["pre_ILI"] > inter_lick_interval_threshold
    licks["bout_end"] = licks["post_ILI"] > inter_lick_interval_threshold
    if licks["bout_start"].sum() != licks["bout_end"].sum():
        raise ValueError("Lick bout splitting failed")
