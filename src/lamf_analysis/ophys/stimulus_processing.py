import numpy as np
import pandas as pd
import pickle

from lamf_analysis.ophys.general_utilities import time_from_last


def _time_col(df):
    """Return a time array from a df that has either 'time' or 'timestamps'."""
    if "time" in df.columns:
        return df["time"].values
    if "timestamps" in df.columns:
        return df["timestamps"].values
    raise KeyError("dataframe has neither 'time' nor 'timestamps' column")


def find_image_changes(image_index: pd.Series, 
                       omitted_index: int) -> np.array:
    '''Find whether each flash was a change flash

    Parameters:
    ----------
    image_index : pd.Series
        Array of image_index of the presented image for each flash
    omitted_index : int
        The index value of the omitted image (often 8)

    Returns:
    -------
    change : np.array of bool
        Whether each flash was a change flash
    '''

    change = np.diff(image_index) != 0
    change = np.concatenate([np.array([False]), change])  # First flash not a change
    omitted = image_index == omitted_index
    omitted_inds = np.flatnonzero(omitted)
    change[omitted_inds] = False

    if image_index.iloc[-1] == omitted_index:
        # If the last flash is omitted we can't set the +1 for that omitted idx
        change[omitted_inds[:-1] + 1] = False
    else:
        change[omitted_inds + 1] = False

    return change

###############################
# Add functions
###############################

def add_prior_image_to_stimulus_presentations(sp_df):
    prior_image_name = [None]
    prior_image_name = prior_image_name + list(sp_df.image_name.values[:-1])
    sp_df['prior_image_name'] = prior_image_name
    return sp_df


def add_licks_to_stimulus_presentations(sp_df, licks, rewards=None,
                                        change_times=None):
    """Add time-from-last-lick / -reward / -change columns to sp_df.

    Parameters
    ----------
    sp_df : pd.DataFrame
        stimulus presentations table (needs 'start_time'; 'is_change' if
        change_times not given).
    licks : pd.DataFrame
        licks table with a 'time' or 'timestamps' column.
    rewards : pd.DataFrame, optional
        rewards table with a 'time' or 'timestamps' column.
    change_times : array-like, optional
        change onset times; if None, derived from sp_df['is_change'].
    """
    flash_times = sp_df["start_time"].values
    lick_times = _time_col(licks)

    if len(lick_times) < 5:  # Passive sessions
        time_from_last_lick = np.full(len(flash_times), np.nan)
    else:
        time_from_last_lick = time_from_last(flash_times, lick_times)

    if rewards is None or len(rewards) < 1:  # Sometimes mice are bad
        time_from_last_reward = np.full(len(flash_times), np.nan)
    else:
        reward_times = _time_col(rewards)
        time_from_last_reward = time_from_last(flash_times, reward_times)

    if change_times is None:
        change_times = sp_df.loc[
            sp_df["is_change"].astype("boolean").fillna(False), "start_time"].values
    change_times = np.asarray(change_times)
    if len(change_times):
        time_from_last_change = time_from_last(flash_times, change_times)
    else:
        time_from_last_change = np.full(len(flash_times), np.nan)

    sp_df["time_from_last_lick"] = time_from_last_lick
    sp_df["time_from_last_reward"] = time_from_last_reward
    sp_df["time_from_last_change"] = time_from_last_change

    return sp_df


def add_stimulus_info_to_stimulus_presentations(sp_df):

    flash_times = sp_df["start_time"].values

    image_indexes = sp_df.groupby("image_name").apply(lambda group: group["image_index"].unique()[0])

    # NOTE: change/omitted already in sp_df, MJD 06/2024
    # if 'omitted' in sp_df['image_name'].unique():
    #     omitted_index = image_indexes['omitted']
    # else:
    #     omitted_index = None

    # changes = find_images_changes(sp_df["image_index"], omitted_index)
    # omitted = sp_df["image_index"] == omitted_index

    # sp_df["is_change"] = changes
    # sp_df["omitted"] = omitted



    # add column: Index of each image block
    changes_including_first = np.copy(sp_df["is_change"].values)
    changes_including_first[0] = True
    change_indices = np.flatnonzero(changes_including_first)
    flash_inds = np.arange(len(sp_df))
    block_inds = np.searchsorted(a=change_indices, v=flash_inds, side="right") - 1
    sp_df["block_index"] = block_inds

    # add column: Block repetition number
    blocks_per_image = sp_df.groupby("image_name").apply(
        lambda group: np.unique(group["block_index"])
    )
    block_repetition_number = np.copy(block_inds)

    for image_name, image_blocks in blocks_per_image.items():
        if image_name != "omitted":
            for ind_block, block_number in enumerate(image_blocks):
                # block_rep_number starts as a copy of block_inds, so we can go write over the index number with the rep number
                block_repetition_number[block_repetition_number == block_number] = ind_block
    sp_df["image_block_repetition"] = block_repetition_number

    # add column: Repeat number within a block
    repeat_number = np.full(len(sp_df), np.nan)
    assert sp_df.iloc[0].name == 0  # Assuming that the row index starts at zero
    for ind_group, group in sp_df.groupby("block_index"):
        repeat = 0
        for ind_row, row in group.iterrows():
            if row["image_name"] != "omitted":
                repeat_number[ind_row] = repeat
                repeat += 1
    sp_df["index_within_block"] = repeat_number

    return sp_df



def trace_average(values, timestamps, t_start, t_end):
    """Mean of a trace over the half-open time window [t_start, t_end).

    Returns np.nan if no samples fall in the window.
    """
    values = np.asarray(values)
    timestamps = np.asarray(timestamps)
    mask = (timestamps >= t_start) & (timestamps < t_end)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(values[mask]))


def add_response_latency(sp_df):
    """Add per-flash licking-response columns using the 'licks' list column.

    Adds: 'licked' (bool), 'rewarded' (bool), 'response_latency' (s from flash
    onset to first lick within the flash window; np.nan if none).
    Requires the 'licks' (and 'rewards') list-per-flash columns added upstream.
    """
    def _latency(row):
        licks = row["licks"]
        if licks is None or len(licks) == 0:
            return np.nan
        return float(np.min(licks) - row["start_time"])

    sp_df["response_latency"] = sp_df.apply(_latency, axis=1)
    sp_df["licked"] = sp_df["licks"].apply(lambda x: x is not None and len(x) > 0)
    if "rewards" in sp_df.columns:
        sp_df["rewarded"] = sp_df["rewards"].apply(lambda x: x is not None and len(x) > 0)
    return sp_df


def annotate_flash_rolling_metrics(sp_df, window_s=320.0,
                                   reward_rate_threshold_per_s=1.0 / 90.0):
    """Add a rolling reward rate and an engagement_state label per flash.

    Reward rate is computed time-based (rewards / second) in a centered
    ``window_s`` window over each flash's start_time, with the denominator
    clipped to the session span so edges are not underestimated (Garrett 2025 /
    AllenSDK convention: engaged when reward rate > 2/3 rewards per minute =
    1/90 rew/s). A flash counts as rewarded if it has >= 1 entry in its 'rewards'
    list column.

    Adds: 'reward_rate' (rew/s), 'engagement_state' ('engaged'/'disengaged').
    """
    t = sp_df["start_time"].values.astype(float)
    if "rewarded" in sp_df.columns:
        rewarded = sp_df["rewarded"].values.astype(float)
    else:
        rewarded = sp_df["rewards"].apply(
            lambda x: 1.0 if (x is not None and len(x) > 0) else 0.0).values
    rew_times = t[rewarded > 0]
    rew_times.sort()
    half = window_s / 2.0
    lo = np.maximum(t - half, t.min())
    hi = np.minimum(t + half, t.max())
    n = np.searchsorted(rew_times, hi, "right") - np.searchsorted(rew_times, lo, "left")
    span = np.clip(hi - lo, 1e-9, None)
    reward_rate = n / span
    sp_df["reward_rate"] = reward_rate
    sp_df["engagement_state"] = np.where(
        reward_rate > reward_rate_threshold_per_s, "engaged", "disengaged")
    return sp_df


def extended_stimulus_presentations_table(sp_df: pd.DataFrame,
                                          licks: pd.DataFrame,
                                          rewards: pd.DataFrame,
                                          change_times: np.array,
                                          running_speed_df: pd.DataFrame,
                                          pupil_area: pd.DataFrame):

    # sp_df = sp_df.copy()
    sp_df = add_prior_image_to_stimulus_presentations(sp_df)
    # ensure a 'change' alias exists (some downstream code uses 'change')
    if "change" not in sp_df.columns and "is_change" in sp_df.columns:
        sp_df["change"] = sp_df["is_change"].astype("boolean").fillna(False).astype(bool)
    sp_df = add_stimulus_info_to_stimulus_presentations(sp_df)

    lick_times = _time_col(licks)
    reward_times = _time_col(rewards)

    # normalize column names from comb (timestamps -> time) so the trace_average
    # calls below work regardless of the caller's schema
    if "time" not in running_speed_df.columns and "timestamps" in running_speed_df.columns:
        running_speed_df = running_speed_df.rename(columns={"timestamps": "time"})
    if pupil_area is not None and "time" not in pupil_area.columns \
            and "timestamps" in pupil_area.columns:
        pupil_area = pupil_area.rename(columns={"timestamps": "time"})
    if "omitted" in sp_df.columns:
        sp_df["omitted"] = sp_df["omitted"].astype("boolean").fillna(False).astype(bool)


    # Lists of licks/rewards on each flash
    licks_each_flash = sp_df.apply(
        lambda row: lick_times[
            ((lick_times > row["start_time"]) & (lick_times < row["start_time"] + 0.75))
        ],
        axis=1,
    )
    rewards_each_flash = sp_df.apply(
        lambda row: reward_times[
            (
                (reward_times > row["start_time"])
                & (reward_times < row["start_time"] + 0.75)
            )
        ],
        axis=1,
    )

    sp_df["licks"] = licks_each_flash
    sp_df["rewards"] = rewards_each_flash

    # Average running speed on each flash
    flash_running_speed = sp_df.apply(
        lambda row: trace_average(
            running_speed_df['speed'].values,
            running_speed_df['time'].values,
            row["start_time"],
            row["start_time"] + 0.25, ), axis=1, )
    sp_df["mean_running_speed"] = flash_running_speed

    # Average running speed before each flash
    pre_flash_running_speed = sp_df.apply(
        lambda row: trace_average(
            running_speed_df['speed'].values,
            running_speed_df['time'].values,
            row["start_time"] - 0.25,
            row["start_time"], ), axis=1, )
    sp_df["pre_flash_running_speed"] = pre_flash_running_speed

    if pupil_area is not None:
        # Average running speed on each flash
        flash_pupil_area = sp_df.apply(
            lambda row: trace_average(
                pupil_area['pupil_area'].values,
                pupil_area['time'].values,
                row["start_time"],
                row["start_time"] + 0.25, ), axis=1, )
        sp_df["mean_pupil_area"] = flash_pupil_area

        # Average running speed before each flash
        pre_flash_pupil_area = sp_df.apply(
            lambda row: trace_average(
                pupil_area['pupil_area'].values,
                pupil_area['time'].values,
                row["start_time"] - 0.25,
                row["start_time"], ), axis=1, )
        sp_df["pre_flash_pupil_area"] = pre_flash_pupil_area

    # add flass after omitted
    sp_df['flash_after_omitted'] = np.hstack((False, sp_df.omitted.values[:-1]))
    sp_df['flash_after_change'] = np.hstack((False, sp_df.change.values[:-1]))
    # add licking responses
    sp_df = add_response_latency(sp_df)

    # sp_df = add_inter_flash_lick_diff_to_stimulus_presentations(sp_df)
    # sp_df = add_first_lick_in_bout_to_stimulus_presentations(sp_df)
    # sp_df = get_consumption_licks(sp_df)
    # sp_df = get_metrics(sp_df, licks, rewards)
    sp_df = annotate_flash_rolling_metrics(sp_df)

    return sp_df


def get_flashes_since_change(
    stimulus_presentations: pd.DataFrame,
) -> pd.Series:
    """Calculate the number of times an images is flashed between changes.

    Parameters
    ----------
    stimulus_presentations : pandas.DataFrame
        Table of presented stimuli with ``is_change`` column already
        calculated.

    Returns
    -------
    flashes_since_change : pandas.Series
        Number of times the same image is flashed between image changes.
    """
    flashes_since_change = pd.Series(
        data=np.zeros(len(stimulus_presentations), dtype=float),
        index=stimulus_presentations.index,
        name="flashes_since_change",
        dtype="int",
    )


    for idx, (pd_index, row) in enumerate(stimulus_presentations.iterrows()):
        omitted = row["omitted"]
        if pd.isna(row["omitted"]):
            omitted = False
        if row["image_name"] == "omitted" or omitted:
            flashes_since_change.iloc[idx] = flashes_since_change.iloc[idx - 1]
        else:
            if row["is_change"] or idx == 0:
                flashes_since_change.iloc[idx] = 0
            else:
                flashes_since_change.iloc[idx] = (
                    flashes_since_change.iloc[idx - 1] + 1
                )
    return flashes_since_change


def get_flashes_since_omission(
    stimulus_presentations: pd.DataFrame,
) -> pd.Series:
    """Calculate the number of times an images is flashed between omissions.

    Parameters
    ----------
    stimulus_presentations : pandas.DataFrame
        Table of presented stimuli with ``is_change`` column already
        calculated.

    Returns
    -------
    flashes_since_omission : pandas.Series
        Number of times the same image is flashed between image omissions.
    """
    flashes_since_omission = pd.Series(
        data=np.zeros(len(stimulus_presentations), dtype=float),
        index=stimulus_presentations.index,
        name="flashes_since_omission",
        dtype="int",
    )

    for idx, (pd_index, row) in enumerate(stimulus_presentations.iterrows()):
        omitted = row["omitted"]
        if pd.isna(row["omitted"]):
            omitted = False
        if (row["image_name"] == "omitted") or (omitted) or idx == 0:
            flashes_since_omission.iloc[idx] = 0
        else:
            flashes_since_omission.iloc[idx] = (
                flashes_since_omission.iloc[idx - 1] + 1
            )
    return flashes_since_omission