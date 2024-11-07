import numpy as np
import pandas as pd
import pickle


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


def add_licks_to_stimulus_presentations(sp_df, licks):
    """
    Add licks to stimulus_presentations_df
    """


    # TODO: passive/active should 
    if len(lick_times) < 5:  # Passive sessions
        time_from_last_lick = np.full(len(flash_times), np.nan)
    else:
        time_from_last_lick = time_from_last(flash_times, lick_times)

    if len(reward_times) < 1:  # Sometimes mice are bad
        time_from_last_reward = np.full(len(flash_times), np.nan)
    else:
        time_from_last_reward = time_from_last(flash_times, reward_times)

    time_from_last_change = time_from_last(flash_times, change_times)

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

    # sp_df["change"] = changes
    # sp_df["omitted"] = omitted



    # add column: Index of each image block
    changes_including_first = np.copy(changes)
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

    for image_name, image_blocks in blocks_per_image.iteritems():
        if image_name != "omitted":
            for ind_block, block_number in enumerate(image_blocks):
                # block_rep_number starts as a copy of block_inds, so we can go write over the index number with the rep number
                block_repetition_number[block_repetition_number == block_number] = ind_block
    sp_df["image_block_repetition"] = block_repetition_number

    # add column: Repeat number within a block
    repeat_number = np.full(len(sp_df), np.nan)
    assert (
        sp_df.iloc[0].name == 0
    )  # Assuming that the row index starts at zero
    for ind_group, group in sp_df.groupby("block_index"):
        repeat = 0
        for ind_row, row in group.iterrows():
            if row["image_name"] != "omitted":
                repeat_number[ind_row] = repeat
                repeat += 1

    sp_df["index_within_block"] = repeat_number

    return sp_df



def extended_stimulus_presentations_table(sp_df,
                                          licks,
                                          rewards,
                                          change_times,
                                          running_speed_df,
                                          pupil_area):
    
    # sp_df = sp_df.copy()
    sp_df = add_prior_image_to_stimulus_presentations(sp_df)
    sp_df = add_stimulus_info_to_stimulus_presentations(sp_df)


    lick_times = licks['time'].values
    reward_times = rewards['time'].values
    # Time from last other for each flash
    




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