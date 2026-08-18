"""Task-performance metrics for change-detection behavior sessions.

The trial log stores stimulus event locations as frame indices. Functions that
accept ``stimulus_timestamps`` expect raw stimulus-vsync times and add
``monitor_delay_s`` to align those frames with displayed-stimulus, lick, and
engagement times. In contrast, ``stimulus_presentations["start_time"]`` is
expected to already include that display correction.
"""

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm


# Default used by comb and ``behavior.stim_table.get_stimulus_presentations``.
MONITOR_DELAY_S = 0.03613
RESPONSE_WINDOW_S = (0.15, 0.75)


def _times(values, name: str, *, sort: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.sort(result) if sort else result


def _engagement_inputs(sample_times, engaged) -> tuple[np.ndarray, np.ndarray]:
    sample_times = _times(sample_times, "engagement_times")
    engaged = np.asarray(engaged)
    if engaged.ndim != 1:
        raise ValueError("engaged must be one-dimensional")
    if sample_times.size != engaged.size:
        raise ValueError("engagement_times and engaged must have equal length")
    if sample_times.size == 0:
        raise ValueError("engagement_times and engaged must not be empty")
    if np.any(np.diff(sample_times) < 0):
        raise ValueError("engagement_times must be sorted")
    if not np.issubdtype(engaged.dtype, np.bool_):
        raise ValueError("engaged must contain boolean values")
    return sample_times, engaged


def _engaged_at(query_times, engagement_times, engaged) -> np.ndarray:
    """Look up the first engagement sample at or after each query time."""
    query_times = np.asarray(query_times, dtype=float)
    indices = np.searchsorted(engagement_times, query_times)
    return engaged[np.clip(indices, 0, engagement_times.size - 1)]


def _event(trial: Mapping, name: str, subtype=None):
    for event in trial.get("events", ()):
        if event[0] == name and (subtype is None or event[1] == subtype):
            return event
    return None


def _event_frame(trial: Mapping, name: str):
    event = _event(trial, name)
    return None if event is None else int(event[3])


def _event_time(trial: Mapping, name: str, subtype=None):
    event = _event(trial, name, subtype)
    return None if event is None else float(event[2])


def _rate(numerator: int, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else np.nan


def signal_detection_metrics(
    hit_rate: float,
    false_alarm_rate: float,
    n_signal_trials: int,
    n_noise_trials: int,
) -> tuple[float, float]:
    """Return loglinear-corrected d-prime and criterion.

    Rates are clipped to ``0.5 / n`` and ``1 - 0.5 / n`` to keep the
    inverse-normal transform finite.

    Parameters
    ----------
    hit_rate
        Probability of a response on signal trials.
    false_alarm_rate
        Probability of a response on noise trials.
    n_signal_trials
        Number of observations used to estimate ``hit_rate``.
    n_noise_trials
        Number of observations used to estimate ``false_alarm_rate``.

    Returns
    -------
    tuple of float
        ``(dprime, criterion)``. Both values are NaN when either rate is
        undefined or its observation count is not positive.
    """

    def clipped(rate, n_trials):
        if not np.isfinite(rate) or n_trials <= 0:
            return np.nan
        lower = 0.5 / n_trials
        return min(max(rate, lower), 1.0 - lower)

    hit_rate = clipped(hit_rate, n_signal_trials)
    false_alarm_rate = clipped(false_alarm_rate, n_noise_trials)
    if not (np.isfinite(hit_rate) and np.isfinite(false_alarm_rate)):
        return np.nan, np.nan
    hit_z = norm.ppf(hit_rate)
    false_alarm_z = norm.ppf(false_alarm_rate)
    return hit_z - false_alarm_z, -0.5 * (hit_z + false_alarm_z)


def trial_performance_metrics(
    trial_log: Sequence[Mapping],
    stimulus_timestamps,
    engagement_times,
    engaged,
    *,
    monitor_delay_s: float = MONITOR_DELAY_S,
) -> dict:
    """Quantify trial outcomes, reward, engagement, and impulsivity.

    Parameters
    ----------
    trial_log
        Camstim trial dictionaries containing ``trial_params``, ``events``,
        and optionally ``rewards``.
    stimulus_timestamps
        Raw stimulus-vsync times indexed by trial event frame. These times do
        not yet include the display delay.
    engagement_times
        Sorted sample times for ``engaged``, on the same clock as licks and
        displayed stimuli.
    engaged
        Boolean engagement state at every ``engagement_times`` sample. String
        labels such as those returned by ``engagement_state`` must first be
        converted explicitly, for example ``state == "engaged"``.
    monitor_delay_s
        Delay from stimulus vsync to displayed stimulus. It is added to raw
        ``stimulus_timestamps`` before event frames are matched to engagement.
        Pass ``0`` only when the supplied timestamps are already display-time
        corrected.

    Returns
    -------
    dict
        Trial counts and outcomes; engaged trial counts; elapsed, engaged, and
        at-risk times; reward count and volume in microliters; hit, catch-FA,
        abort-FA, and premature-response rates; and catch-based d-prime and
        criterion. Rate and SDT values are NaN when their denominators are
        zero.

    Notes
    -----
    Autorewarded go trials are excluded from hit, abort, and at-risk metrics.
    Catch trials contribute only to catch-FA metrics. Engagement is sampled at
    the (sham-)change for completed trials and at the abort for aborted trials.
    ``engaged_time_min`` assumes a uniformly sampled engagement grid and uses
    its median sample interval.
    """
    stimulus_timestamps = _times(stimulus_timestamps, "stimulus_timestamps")
    engagement_times, engaged = _engagement_inputs(engagement_times, engaged)
    if monitor_delay_s < 0:
        raise ValueError("monitor_delay_s must be non-negative")

    def engaged_at_frame(frame):
        if frame is None or frame < 0 or frame >= stimulus_timestamps.size:
            return None
        event_time = stimulus_timestamps[frame] + monitor_delay_s
        return bool(_engaged_at(event_time, engagement_times, engaged))

    n_go = n_catch = n_autoreward = n_go_scored = n_abort_go = 0
    at_risk_time = 0.0
    go_outcomes = []
    go_engaged = []
    abort_outcomes = []
    abort_engaged = []
    catch_outcomes = []
    catch_engaged = []

    # Classify each trial once; go, catch, and autoreward populations have
    # intentionally different denominators.
    for trial in trial_log:
        params = trial.get("trial_params", {})
        is_catch = bool(params.get("catch", False))
        is_autoreward = bool(params.get("auto_reward", False))
        n_catch += int(is_catch)
        n_go += int(not is_catch)
        n_autoreward += int(is_autoreward)

        # Catch FA is scored only when a catch trial reaches the sham change.
        if is_catch:
            engagement = engaged_at_frame(_event_frame(trial, "sham_change"))
            if engagement is not None:
                if _event(trial, "false_alarm") is not None:
                    catch_outcomes.append(True)
                    catch_engaged.append(engagement)
                elif _event(trial, "rejection") is not None:
                    catch_outcomes.append(False)
                    catch_engaged.append(engagement)
            continue
        if is_autoreward:
            continue

        # Abort FA is scored on non-autoreward go trials, including aborts.
        n_go_scored += 1
        aborted = _event(trial, "abort") is not None
        start_time = _event_time(trial, "pre_change", "enter")
        if start_time is None:
            start_time = _event_time(trial, "trial_start")
        if aborted:
            n_abort_go += 1
            end_time = _event_time(trial, "abort")
            reference_frame = _event_frame(trial, "abort")
        else:
            end_time = _event_time(trial, "stimulus_changed")
            reference_frame = _event_frame(trial, "stimulus_changed")
        if start_time is not None and end_time is not None and end_time >= start_time:
            at_risk_time += end_time - start_time

        engagement = engaged_at_frame(reference_frame)
        abort_outcomes.append(aborted)
        abort_engaged.append(bool(engagement) if engagement is not None else False)
        # Hit rate is scored only on completed, non-autoreward go trials.
        if not aborted:
            if _event(trial, "hit") is not None:
                go_outcomes.append(True)
                go_engaged.append(bool(engagement))
            elif _event(trial, "miss") is not None:
                go_outcomes.append(False)
                go_engaged.append(bool(engagement))

    go_outcomes = np.asarray(go_outcomes, dtype=bool)
    go_engaged = np.asarray(go_engaged, dtype=bool)
    abort_outcomes = np.asarray(abort_outcomes, dtype=bool)
    abort_engaged = np.asarray(abort_engaged, dtype=bool)
    catch_outcomes = np.asarray(catch_outcomes, dtype=bool)
    catch_engaged = np.asarray(catch_engaged, dtype=bool)

    def outcomes(values, mask):
        positive = int(values[mask].sum())
        negative = int(mask.sum()) - positive
        return positive, negative, _rate(positive, positive + negative)

    all_go = np.ones(go_outcomes.size, dtype=bool)
    all_catch = np.ones(catch_outcomes.size, dtype=bool)
    hit, miss, hit_rate = outcomes(go_outcomes, all_go)
    fa, cr, catch_fa = outcomes(catch_outcomes, all_catch)
    hit_eng, miss_eng, hit_rate_eng = outcomes(go_outcomes, go_engaged)
    fa_eng, cr_eng, catch_fa_eng = outcomes(catch_outcomes, catch_engaged)
    n_abort_engaged = int(abort_engaged.sum())
    abort_fa_engaged = (
        _rate(int(abort_outcomes[abort_engaged].sum()), n_abort_engaged)
        if n_abort_engaged
        else np.nan
    )
    dprime, criterion = signal_detection_metrics(
        hit_rate, catch_fa, hit + miss, fa + cr
    )
    dprime_engaged, criterion_engaged = signal_detection_metrics(
        hit_rate_eng, catch_fa_eng, hit_eng + miss_eng, fa_eng + cr_eng
    )

    # Camstim stores reward volume in mL; report microliters for compatibility
    # with the existing behavior reports.
    reward_ul = 0.0
    n_rewards = 0
    for trial in trial_log:
        for reward in trial.get("rewards") or ():
            if reward and reward[0] is not None:
                reward_ul += float(reward[0]) * 1000.0
                n_rewards += 1

    session_duration_min = (
        float(engagement_times[-1] - engagement_times[0]) / 60.0
    )
    engaged_fraction = float(engaged.mean())
    engagement_sample_interval_s = (
        float(np.median(np.diff(engagement_times)))
        if engagement_times.size > 1
        else 0.0
    )

    return {
        "n_trials": len(trial_log),
        "n_go": n_go,
        "n_catch": n_catch,
        "n_autoreward": n_autoreward,
        "hit": hit,
        "miss": miss,
        "false_alarm": fa,
        "correct_reject": cr,
        "at_risk_time_s": at_risk_time,
        "n_go_engaged": hit_eng + miss_eng,
        "n_catch_engaged": fa_eng + cr_eng,
        "n_go_abort_engaged": n_abort_engaged,
        "n_engaged_trials": int(go_engaged.sum() + catch_engaged.sum()),
        "engaged_fraction": engaged_fraction,
        "engaged_time_min": (
            float(engaged.sum()) * engagement_sample_interval_s / 60.0
        ),
        "session_duration_min": session_duration_min,
        "reward_ul": reward_ul,
        "n_rewards": n_rewards,
        "abort_false_alarm_rate": _rate(int(abort_outcomes.sum()), n_go_scored),
        "abort_false_alarm_rate_engaged": abort_fa_engaged,
        "premature_rate_per_s": _rate(n_abort_go, at_risk_time),
        "hit_rate": hit_rate,
        "catch_false_alarm_rate": catch_fa,
        "dprime_catch": dprime,
        "criterion_catch": criterion,
        "hit_rate_engaged": hit_rate_eng,
        "catch_false_alarm_rate_engaged": catch_fa_eng,
        "dprime_catch_engaged": dprime_engaged,
        "criterion_catch_engaged": criterion_engaged,
    }


def per_flash_false_alarm_rate(
    stimulus_presentations: pd.DataFrame,
    lick_times,
    engagement_times,
    engaged,
    *,
    response_window_s: tuple[float, float] = RESPONSE_WINDOW_S,
    min_clean_presentations: int = 20,
    min_session_presentations: int = 1000,
) -> dict:
    """Compute false-alarm rate on clean repeated flashes.

    Parameters
    ----------
    stimulus_presentations
        Presentation table with ``start_time`` (already display-delay
        corrected), ``omitted``, ``is_change``, and ``flashes_since_change``.
    lick_times
        Lick timestamps on the same clock as presentation start times.
    engagement_times, engaged
        Sorted engagement sample times and corresponding boolean states.
    response_window_s
        Inclusive lick-response window relative to each flash onset.
    min_clean_presentations
        Minimum eligible observations required to report each rate.
    min_session_presentations
        Sessions at or below this presentation count are treated as
        continuous-grating sessions without discrete flash structure.

    Returns
    -------
    dict
        Ungated and engaged false-alarm rates plus counts of eligible clean
        presentations. A rate is NaN when its count is below threshold.

    Notes
    -----
    A clean presentation is a non-change, non-omitted flash that does not
    immediately follow an omission and is at least four flashes after a
    change. This excludes change/reward-adjacent consummatory responses.
    """
    required = {"start_time", "omitted", "is_change", "flashes_since_change"}
    missing = required.difference(stimulus_presentations.columns)
    if missing:
        raise ValueError(f"stimulus_presentations missing columns: {sorted(missing)}")
    if min_clean_presentations <= 0 or min_session_presentations < 0:
        raise ValueError("presentation count thresholds must be non-negative")
    if len(stimulus_presentations) <= min_session_presentations:
        return {
            "false_alarm_rate": np.nan,
            "false_alarm_rate_engaged": np.nan,
            "n_clean": 0,
            "n_clean_engaged": 0,
        }

    lick_times = _times(lick_times, "lick_times", sort=True)
    engagement_times, engaged = _engagement_inputs(engagement_times, engaged)
    omitted = (
        stimulus_presentations["omitted"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    is_change = (
        stimulus_presentations["is_change"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    flashes_since_change = stimulus_presentations[
        "flashes_since_change"
    ].to_numpy(dtype=float)
    previous_omitted = np.r_[False, omitted[:-1]]
    # Drop change, omission, post-omission, and post-change/reward flashes.
    clean = (
        ~is_change
        & ~omitted
        & ~previous_omitted
        & (flashes_since_change >= 4)
    )
    start_times = _times(stimulus_presentations["start_time"], "start_time")
    responded = _responses(lick_times, start_times, response_window_s)
    engaged_flash = _engaged_at(start_times, engagement_times, engaged)
    clean_engaged = clean & engaged_flash
    return {
        "false_alarm_rate": (
            float(responded[clean].mean())
            if clean.sum() >= min_clean_presentations
            else np.nan
        ),
        "false_alarm_rate_engaged": (
            float(responded[clean_engaged].mean())
            if clean_engaged.sum() >= min_clean_presentations
            else np.nan
        ),
        "n_clean": int(clean.sum()),
        "n_clean_engaged": int(clean_engaged.sum()),
    }


def matched_flash_false_alarm_rate(
    stimulus_presentations: pd.DataFrame,
    lick_times,
    engagement_times,
    engaged,
    *,
    response_window_s: tuple[float, float] = RESPONSE_WINDOW_S,
    min_clean_presentations: int = 20,
    min_session_presentations: int = 1000,
) -> dict:
    """Compute repeat-flash FA matched to the change-position distribution.

    Parameters
    ----------
    stimulus_presentations
        Presentation table described by :func:`per_flash_false_alarm_rate`.
    lick_times
        Lick timestamps on the presentation time base.
    engagement_times, engaged
        Sorted engagement sample times and corresponding boolean states.
    response_window_s
        Inclusive lick-response window relative to each flash onset.
    min_clean_presentations
        Minimum clean observations required to report a rate.
    min_session_presentations
        Minimum size used to identify sessions with discrete flashes.

    Returns
    -------
    dict
        Ungated and engaged matched false-alarm rates. Rates are NaN when
        there are too few clean observations or no usable change positions.

    Notes
    -----
    The response probability at each clean ``flashes_since_change`` position
    is weighted by how often real changes occurred at that position.
    """
    base = per_flash_false_alarm_rate(
        stimulus_presentations,
        lick_times,
        engagement_times,
        engaged,
        response_window_s=response_window_s,
        min_clean_presentations=min_clean_presentations,
        min_session_presentations=min_session_presentations,
    )
    if base["n_clean"] == 0:
        return {
            "false_alarm_rate": np.nan,
            "false_alarm_rate_engaged": np.nan,
        }

    lick_times = _times(lick_times, "lick_times", sort=True)
    engagement_times, engaged = _engagement_inputs(engagement_times, engaged)
    omitted = (
        stimulus_presentations["omitted"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    is_change = (
        stimulus_presentations["is_change"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    flashes_since_change = stimulus_presentations[
        "flashes_since_change"
    ].to_numpy(dtype=float)
    clean = (
        ~is_change
        & ~omitted
        & ~np.r_[False, omitted[:-1]]
        & (flashes_since_change >= 4)
    )
    start_times = _times(stimulus_presentations["start_time"], "start_time")
    responded = _responses(lick_times, start_times, response_window_s)
    engaged_flash = _engaged_at(start_times, engagement_times, engaged)
    # Reweight clean repeats to the empirical change-position distribution.
    change_indices = np.flatnonzero(is_change)
    change_indices = change_indices[change_indices > 0]
    change_positions = (flashes_since_change[change_indices - 1] + 1).astype(int)
    weights = Counter(change_positions)

    def matched(mask):
        numerator = denominator = 0.0
        for position, weight in weights.items():
            selected = mask & (flashes_since_change == position)
            if selected.any():
                numerator += weight * responded[selected].mean()
                denominator += weight
        return numerator / denominator if denominator else np.nan

    return {
        "false_alarm_rate": (
            float(matched(clean))
            if clean.sum() >= min_clean_presentations
            else np.nan
        ),
        "false_alarm_rate_engaged": (
            float(matched(clean & engaged_flash))
            if (clean & engaged_flash).sum() >= min_clean_presentations
            else np.nan
        ),
    }


def per_time_false_alarm_rate(
    trial_log: Sequence[Mapping],
    stimulus_timestamps,
    lick_times,
    engagement_times,
    engaged,
    *,
    response_window_s: tuple[float, float] = RESPONSE_WINDOW_S,
    step_s: float = 0.75,
    pre_change_exclusion_s: float = 0.4,
    post_change_exclusion_s: float = 2.25,
    monitor_delay_s: float = MONITOR_DELAY_S,
    min_changes: int = 5,
    min_samples: int = 20,
) -> dict:
    """Compute flash-free FA on a regular time grid.

    Parameters
    ----------
    trial_log
        Camstim trial dictionaries containing real and sham change events.
    stimulus_timestamps
        Raw stimulus-vsync times indexed by trial event frame.
    lick_times
        Lick timestamps on the displayed-stimulus time base.
    engagement_times, engaged
        Sorted engagement sample times and corresponding boolean states.
    response_window_s
        Inclusive lick-response window relative to each time-grid sample.
    step_s
        Spacing of the regular false-alarm sampling grid.
    pre_change_exclusion_s, post_change_exclusion_s
        Time removed before and after every real or sham change.
    monitor_delay_s
        Delay added to raw stimulus-vsync times when locating changes on the
        lick and engagement clock.
    min_changes
        Minimum number of real plus sham changes required.
    min_samples
        Minimum eligible samples required for each reported rate.

    Returns
    -------
    dict
        Ungated and engaged false-alarm rates and their eligible sample
        counts. Rates are NaN when the relevant minimum is not met.

    Notes
    -----
    This metric does not require discrete flashes, so it is defined for
    continuous-grating sessions. Grid points near changes are excluded to
    approximate the clean-flash exclusions used by per-flash FA.
    """
    stimulus_timestamps = _times(stimulus_timestamps, "stimulus_timestamps")
    lick_times = _times(lick_times, "lick_times", sort=True)
    engagement_times, engaged = _engagement_inputs(engagement_times, engaged)
    if monitor_delay_s < 0:
        raise ValueError("monitor_delay_s must be non-negative")
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    if pre_change_exclusion_s < 0 or post_change_exclusion_s < 0:
        raise ValueError("change exclusion windows must be non-negative")
    if min_changes <= 0 or min_samples <= 0:
        raise ValueError("min_changes and min_samples must be positive")

    # Trial events are frame-indexed; convert their raw vsync times to display
    # times before comparing them with licks and engagement samples.
    change_times = []
    for trial in trial_log:
        for name in ("stimulus_changed", "sham_change"):
            frame = _event_frame(trial, name)
            if frame is not None and 0 <= frame < stimulus_timestamps.size:
                change_times.append(stimulus_timestamps[frame] + monitor_delay_s)
    change_times = np.sort(np.asarray(change_times, dtype=float))
    if change_times.size < min_changes or lick_times.size == 0:
        return {
            "false_alarm_rate": np.nan,
            "false_alarm_rate_engaged": np.nan,
            "n_samples": 0,
            "n_samples_engaged": 0,
        }

    samples = np.arange(change_times.min() - 1.0, change_times.max() + 3.0, step_s)
    # Remove samples whose response windows could overlap change-related
    # behavior, mirroring the post-change exclusion in per-flash FA.
    near_change = (
        np.searchsorted(
            change_times, samples + pre_change_exclusion_s, side="right"
        )
        - np.searchsorted(
            change_times, samples - post_change_exclusion_s, side="right"
        )
    ) > 0
    samples = samples[~near_change]
    if samples.size < min_samples:
        return {
            "false_alarm_rate": np.nan,
            "false_alarm_rate_engaged": np.nan,
            "n_samples": int(samples.size),
            "n_samples_engaged": 0,
        }

    responded = _responses(lick_times, samples, response_window_s)
    engaged_samples = _engaged_at(samples, engagement_times, engaged)
    n_engaged = int(engaged_samples.sum())
    return {
        "false_alarm_rate": float(responded.mean()),
        "false_alarm_rate_engaged": (
            float(responded[engaged_samples].mean())
            if n_engaged >= min_samples
            else np.nan
        ),
        "n_samples": int(samples.size),
        "n_samples_engaged": n_engaged,
    }


def task_performance_metrics(
    trial_log: Sequence[Mapping],
    stimulus_timestamps,
    stimulus_presentations: pd.DataFrame,
    lick_times,
    engagement_times,
    engaged,
    *,
    monitor_delay_s: float = MONITOR_DELAY_S,
) -> dict:
    """Return all task-performance metrics for one session.

    Parameters
    ----------
    trial_log
        Camstim trial dictionaries.
    stimulus_timestamps
        Raw stimulus-vsync times indexed by trial event frame.
    stimulus_presentations
        Discrete presentation table whose ``start_time`` is already corrected
        to displayed-stimulus time.
    lick_times
        Lick timestamps on the displayed-stimulus time base.
    engagement_times, engaged
        Sorted engagement sample times and corresponding boolean states.
    monitor_delay_s
        Delay added only to raw ``stimulus_timestamps``. It is not added to
        ``stimulus_presentations["start_time"]``.

    Returns
    -------
    dict
        All outputs from :func:`trial_performance_metrics`, the per-flash,
        matched, and per-time false-alarm rates and sample counts, and d-prime
        and criterion derived from per-flash and per-time FA in ungated and
        engagement-gated populations.
    """
    # Trial metrics establish the hit-rate side of each SDT comparison.
    result = trial_performance_metrics(
        trial_log,
        stimulus_timestamps,
        engagement_times,
        engaged,
        monitor_delay_s=monitor_delay_s,
    )
    per_flash = per_flash_false_alarm_rate(
        stimulus_presentations, lick_times, engagement_times, engaged
    )
    matched = matched_flash_false_alarm_rate(
        stimulus_presentations, lick_times, engagement_times, engaged
    )
    per_time = per_time_false_alarm_rate(
        trial_log,
        stimulus_timestamps,
        lick_times,
        engagement_times,
        engaged,
        monitor_delay_s=monitor_delay_s,
    )
    result.update(
        per_flash_false_alarm_rate=per_flash["false_alarm_rate"],
        per_flash_false_alarm_rate_engaged=per_flash[
            "false_alarm_rate_engaged"
        ],
        n_clean_presentations=per_flash["n_clean"],
        n_clean_presentations_engaged=per_flash["n_clean_engaged"],
        matched_false_alarm_rate=matched["false_alarm_rate"],
        matched_false_alarm_rate_engaged=matched[
            "false_alarm_rate_engaged"
        ],
        per_time_false_alarm_rate=per_time["false_alarm_rate"],
        per_time_false_alarm_rate_engaged=per_time[
            "false_alarm_rate_engaged"
        ],
        n_time_samples=per_time["n_samples"],
        n_time_samples_engaged=per_time["n_samples_engaged"],
    )
    # Pair each FA estimate with the hit rate and observation counts from the
    # same ungated or engagement-gated population.
    for suffix, fa_rate, n_noise in (
        (
            "per_flash",
            result["per_flash_false_alarm_rate"],
            result["n_clean_presentations"],
        ),
        (
            "per_flash_engaged",
            result["per_flash_false_alarm_rate_engaged"],
            result["n_clean_presentations_engaged"],
        ),
        (
            "per_time",
            result["per_time_false_alarm_rate"],
            result["n_time_samples"],
        ),
        (
            "per_time_engaged",
            result["per_time_false_alarm_rate_engaged"],
            result["n_time_samples_engaged"],
        ),
    ):
        engaged_metric = suffix.endswith("_engaged")
        hit_rate = (
            result["hit_rate_engaged"] if engaged_metric else result["hit_rate"]
        )
        n_signal = (
            result["n_go_engaged"]
            if engaged_metric
            else result["hit"] + result["miss"]
        )
        dprime, criterion = signal_detection_metrics(
            hit_rate, fa_rate, n_signal, n_noise
        )
        result[f"dprime_{suffix}"] = dprime
        result[f"criterion_{suffix}"] = criterion
    return result


def _responses(lick_times, event_times, response_window_s) -> np.ndarray:
    start, stop = response_window_s
    if start < 0 or stop <= start:
        raise ValueError(
            "response_window_s must contain non-negative increasing values"
        )
    return (
        np.searchsorted(lick_times, event_times + stop, side="right")
        - np.searchsorted(lick_times, event_times + start, side="left")
    ) > 0
