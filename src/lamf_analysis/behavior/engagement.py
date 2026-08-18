"""Time-based reward- and lick-based engagement metrics."""

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d

from lamf_analysis.behavior.licks import get_lick_bout_start_times


EngagementMethod = Literal["reward", "lick"]


def _validated_times(values, name: str, *, sort: bool = False) -> np.ndarray:
    times = np.asarray(values, dtype=float)
    if times.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{name} must contain only finite values")
    return np.sort(times) if sort else times


def reward_rate(
    sample_times,
    reward_times,
    window_s: float = 320.0,
) -> np.ndarray:
    """Compute rewards per second in a centered, edge-clipped time window."""
    sample_times = _validated_times(sample_times, "sample_times")
    reward_times = _validated_times(reward_times, "reward_times", sort=True)
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if sample_times.size == 0:
        return np.array([], dtype=float)

    half_window = window_s / 2.0
    session_start = sample_times.min()
    session_end = sample_times.max()
    lower = np.maximum(sample_times - half_window, session_start)
    upper = np.minimum(sample_times + half_window, session_end)
    reward_count = (
        np.searchsorted(reward_times, upper, side="right")
        - np.searchsorted(reward_times, lower, side="left")
    )
    duration = np.clip(upper - lower, np.finfo(float).eps, None)
    return reward_count / duration


def lick_bout_rate(
    sample_times,
    lick_times,
    gaussian_sd_s: float = 30.0,
    inter_lick_interval_threshold_s: float = 0.7,
    excluded_bout_windows=None,
) -> np.ndarray:
    """Compute a centered Gaussian lick-bout rate in bouts per minute.

    Lick-bout starts are binned on a uniform grid and smoothed with a centered
    Gaussian using nearest edge extension, matching the session01/session03
    detector. Uniform query times are used directly; irregular query times
    (such as flash onsets) are evaluated by interpolation from a one-second
    internal grid.
    """
    sample_times = _validated_times(sample_times, "sample_times")
    lick_times = _validated_times(lick_times, "lick_times")
    if gaussian_sd_s <= 0:
        raise ValueError("gaussian_sd_s must be positive")
    if sample_times.size == 0:
        return np.array([], dtype=float)
    if np.any(np.diff(sample_times) <= 0):
        raise ValueError("sample_times must be strictly increasing")

    if sample_times.size == 1:
        rate_grid = np.array([sample_times[0], sample_times[0] + 1.0])
        return lick_bout_rate(
            rate_grid,
            lick_times,
            gaussian_sd_s=gaussian_sd_s,
            inter_lick_interval_threshold_s=(
                inter_lick_interval_threshold_s
            ),
            excluded_bout_windows=excluded_bout_windows,
        )[:1]

    intervals = np.diff(sample_times)
    is_uniform = np.allclose(
        intervals, intervals[0], rtol=1e-6, atol=1e-9
    )
    if not is_uniform:
        rate_grid = np.arange(
            sample_times[0],
            sample_times[-1] + 1.0,
            1.0,
        )
        grid_rate = lick_bout_rate(
            rate_grid,
            lick_times,
            gaussian_sd_s=gaussian_sd_s,
            inter_lick_interval_threshold_s=(
                inter_lick_interval_threshold_s
            ),
            excluded_bout_windows=excluded_bout_windows,
        )
        return np.interp(sample_times, rate_grid, grid_rate)

    sample_interval_s = intervals[0]

    bout_times = get_lick_bout_start_times(
        lick_times,
        inter_lick_interval_threshold=inter_lick_interval_threshold_s,
    )
    if excluded_bout_windows is not None:
        excluded = np.asarray(excluded_bout_windows, dtype=float)
        if excluded.ndim != 2 or excluded.shape[1] != 2:
            raise ValueError("excluded_bout_windows must have shape (n, 2)")
        keep = np.ones(bout_times.size, dtype=bool)
        for start, stop in excluded:
            if stop < start:
                raise ValueError(
                    "excluded bout window stop must not precede start"
                )
            keep &= ~((bout_times >= start) & (bout_times <= stop))
        bout_times = bout_times[keep]
    if bout_times.size == 0:
        return np.zeros(sample_times.size, dtype=float)

    edges = np.append(
        sample_times - sample_interval_s / 2.0,
        sample_times[-1] + sample_interval_s / 2.0,
    )
    counts, _ = np.histogram(bout_times, bins=edges)
    rate_per_s = gaussian_filter1d(
        counts.astype(float),
        sigma=gaussian_sd_s / sample_interval_s,
        mode="nearest",
    ) / sample_interval_s
    return rate_per_s * 60.0


def consumed_auto_reward_times(
    lick_times,
    auto_reward_times,
    consummatory_window_s: float = 5.0,
) -> np.ndarray:
    """Return auto rewards followed by a lick within the given window."""
    lick_times = _validated_times(lick_times, "lick_times", sort=True)
    auto_reward_times = _validated_times(
        auto_reward_times, "auto_reward_times", sort=True
    )
    if consummatory_window_s <= 0:
        raise ValueError("consummatory_window_s must be positive")

    first_lick = np.searchsorted(lick_times, auto_reward_times, side="right")
    last_lick = np.searchsorted(
        lick_times,
        auto_reward_times + consummatory_window_s,
        side="right",
    )
    return auto_reward_times[last_lick > first_lick]


def consumed_auto_reward_mask(
    sample_times,
    lick_times,
    auto_reward_times,
    post_reward_window_s: float = 10.0,
    consummatory_window_s: float = 5.0,
) -> np.ndarray:
    """Identify post-auto-reward periods with consummatory licking.

    An auto reward is considered consumed when at least one lick occurs from
    reward delivery through ``post_reward_window_s`` seconds afterward. Only
    consumed auto rewards contribute to the returned mask.
    """
    sample_times = _validated_times(sample_times, "sample_times")
    if post_reward_window_s <= 0:
        raise ValueError("post_reward_window_s must be positive")

    consumed_rewards = consumed_auto_reward_times(
        lick_times,
        auto_reward_times,
        consummatory_window_s=consummatory_window_s,
    )
    mask = np.zeros(sample_times.size, dtype=bool)
    for reward_time in consumed_rewards:
        mask |= (
            (sample_times >= reward_time)
            & (sample_times <= reward_time + post_reward_window_s)
        )
    return mask


def engagement_state(
    sample_times,
    *,
    method: EngagementMethod,
    reward_times=None,
    lick_times=None,
    auto_reward_times=None,
    reward_window_s: float = 320.0,
    reward_rate_threshold_per_s: float = 1.0 / 90.0,
    lick_gaussian_sd_s: float = 30.0,
    lick_rate_threshold_per_min: float = 3.0,
    inter_lick_interval_threshold_s: float = 0.7,
    auto_reward_post_window_s: float = 10.0,
    auto_reward_consummatory_window_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected engagement rate and state at each sample time.

    States are ``"engaged"`` and ``"disengaged"``. For lick-based
    engagement, samples in a consumed auto-reward window are labeled
    ``"auto_reward"`` when ``auto_reward_times`` are supplied.
    """
    sample_times = _validated_times(sample_times, "sample_times")

    if method == "reward":
        if reward_times is None:
            raise ValueError("reward_times are required for reward engagement")
        rate = reward_rate(sample_times, reward_times, window_s=reward_window_s)
        state = np.where(
            rate > reward_rate_threshold_per_s, "engaged", "disengaged"
        ).astype(object)
        return rate, state

    if method == "lick":
        if lick_times is None:
            raise ValueError("lick_times are required for lick engagement")
        consumed_rewards = (
            consumed_auto_reward_times(
                lick_times,
                auto_reward_times,
                consummatory_window_s=auto_reward_consummatory_window_s,
            )
            if auto_reward_times is not None
            else np.array([], dtype=float)
        )
        excluded_bout_windows = (
            np.column_stack(
                (
                    consumed_rewards,
                    consumed_rewards + auto_reward_post_window_s,
                )
            )
            if consumed_rewards.size
            else None
        )
        rate = lick_bout_rate(
            sample_times,
            lick_times,
            gaussian_sd_s=lick_gaussian_sd_s,
            inter_lick_interval_threshold_s=inter_lick_interval_threshold_s,
            excluded_bout_windows=excluded_bout_windows,
        )
        state = np.where(
            rate > lick_rate_threshold_per_min, "engaged", "disengaged"
        ).astype(object)
        if auto_reward_times is not None:
            mask = consumed_auto_reward_mask(
                sample_times,
                lick_times,
                auto_reward_times,
                post_reward_window_s=auto_reward_post_window_s,
                consummatory_window_s=(
                    auto_reward_consummatory_window_s
                ),
            )
            state[mask] = "autoreward"
        return rate, state

    raise ValueError("method must be 'reward' or 'lick'")
