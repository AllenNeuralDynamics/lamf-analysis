import numpy as np
import pandas as pd
import scipy

import lamf_analysis.utils as lamf_utils


def get_roi_df_with_valid_roi(bod, small_roi_radius_threshold_in_um=4):
    ''' Get roi_df with valid_roi column
    This also mutates bod.cell_specimen_table, so need to be run only once per bod loading

    Parameters
    ----------
    bod : BehaviorOphysDataset
        The behavior ophys dataset object.
    small_roi_radius_threshold_in_um : float
        The threshold for small roi in um^2

    Returns
    -------
    cell_specimen_table : pd.DataFrame
        The cell_specimen_table with valid_roi column
    '''
    cell_specimen_table = bod.cell_specimen_table
    if np.array([k in cell_specimen_table.columns for k in ['touching_motion_border', 'small_roi', 'valid_roi']]).all():
        return cell_specimen_table
    else:
        plane_path = bod.metadata['plane']['plane_path']
        range_y, range_x = lamf_utils.get_motion_correction_crop_xy_range(plane_path)
        range_y = [int(range_y[0]), -int(range_y[1])]
        range_x = [int(range_x[0]), -int(range_x[1])]
        
        on_mask = np.zeros((bod.metadata['plane']['fov_height'], bod.metadata['plane']['fov_width']), dtype=bool)
        on_mask[range_y[0]:range_y[1], range_x[0]:range_x[1]] = True
        motion_mask = ~on_mask

        def _touching_motion_border(row, motion_mask):
            if (row.mask_matrix * motion_mask).any():
                return True
            else:
                return False

        cell_specimen_table['touching_motion_border'] = cell_specimen_table.apply(_touching_motion_border, axis=1, motion_mask=motion_mask)
        
        small_roi_radius_threshold_in_pix = small_roi_radius_threshold_in_um / float(bod.metadata['plane']['fov_scale_factor'])
        area_threshold = np.pi * (small_roi_radius_threshold_in_pix**2)
        
        cell_specimen_table['small_roi'] = cell_specimen_table['mask_matrix'].apply(lambda x: len(np.where(x)[0]) < area_threshold)
        cell_specimen_table['valid_roi'] = ~cell_specimen_table['touching_motion_border'] & ~cell_specimen_table['small_roi']
    
    return cell_specimen_table


def merge_trials_to_stim_table(bod):
    ''' To add hits and misses to stim_table.
    Non-change stimulus presentations will be False.
    This mutates bod.stimulus_presentations, so need to be run only once per bod loading
    '''
    # check if bod has trials
    if not hasattr(bod, 'trials'):
        bod = add_trials_to_bod(bod)
    trials = bod.trials
    stim_table = bod.stimulus_presentations
    stim_table['is_change'] = stim_table.is_change.astype(bool)
    assert np.array_equal(stim_table.query('is_change').start_time.values, trials.change_time.values)
    stim_table['hit'] = False
    stim_table['miss'] = False
    stim_table.loc[stim_table.start_time.isin(trials.query('hit').change_time.values), 'hit'] = True
    stim_table.loc[stim_table.start_time.isin(trials.query('miss').change_time.values), 'miss'] = True

    assert np.all(bod.stimulus_presentations.query('is_change').hit.values == trials.hit.values)
    return stim_table


def add_trials_to_bod(bod, response_window=(0.15, 0.75)):
    """ Temporary fix to add trials to bod
    using stimulus_presentations and licks.
    No correct rejection for this.
    Columns: 'change_time', 'hit', 'miss'

    Parameters
    ----------
    bod : BehaviorOphysDataset
        The behavior ophys dataset object.

    Returns
    -------
    bod : BehaviorOphysDataset
        The behavior ophys dataset object with trials.
    """

    stimulus_presentations = bod.stimulus_presentations
    lick_times = bod.licks.timestamps.values
    trials = pd.DataFrame(columns=['change_time', 'hit', 'miss'])

    stimulus_presentations['is_change'] = stimulus_presentations['is_change'].astype(bool)
    change_times = stimulus_presentations.query('is_change').start_time.values
    response_windows = np.array([change_times + response_window[0], change_times + response_window[1]]).T
    hit = np.zeros(len(change_times), 'bool')
    for i, window in enumerate(response_windows):
        if np.any((lick_times > window[0]) & (lick_times < window[1])):
            hit[i] = 1
    miss = ~hit
    trials = pd.DataFrame({'change_time': change_times, 'hit': hit, 'miss': miss})

    bod.trials = trials
    return bod


# redundant function from GLM design_matrix_tools.py
def get_pupil_area(bod, ophys_timestamps):
    '''
        New eye_tracking results have bad frames. Use these to filter out
        Use area instead of radius. No need to calculate radius from area.
    '''    

    # Set parameters for blink detection, and load data
    eye_df = bod.eye_tracking_table.copy(deep=True)
    pupil_area_df = eye_df.query('eye_is_bad_frame==False and pupil_is_bad_frame==False')[['timestamps', 'pupil_area']].copy()

    # Interpolate everything onto ophys_timestamps
    ophys_eye = pd.DataFrame({'timestamps':ophys_timestamps})
    f = scipy.interpolate.interp1d(pupil_area_df['timestamps'], pupil_area_df['pupil_area'], bounds_error=False)
    ophys_eye['pupil_area'] = f(ophys_eye['timestamps'])
    ophys_eye['pupil_area'] = ophys_eye['pupil_area'].ffill()
    ophys_eye['pupil_area_zscore'] = scipy.stats.zscore(ophys_eye['pupil_area'],nan_policy='omit')
    return ophys_eye


def get_running_speed(bod, ophys_timestamps):
    ''' Running speed interpolated to ophys_timestamps
    '''
    # Set parameters for blink detection, and load data
    running_speed_df = bod.running_speed.copy(deep=True)
    # Interpolate everything onto ophys_timestamps
    running_speed = pd.DataFrame({'timestamps':ophys_timestamps})
    f = scipy.interpolate.interp1d(running_speed_df['timestamps'], running_speed_df['speed'], bounds_error=False)
    running_speed['running_speed'] = f(running_speed['timestamps'])
    running_speed['running_speed'] = running_speed['running_speed'].ffill()
    running_speed['running_speed_zscore'] = scipy.stats.zscore(running_speed['running_speed'],nan_policy='omit')
    return running_speed


# redundant function from GLM design_matrix_tools.py
def interpolate_to_ophys_timestamps(ophys_timestamps, df):
    """ Interpolate timeseries onto ophys timestamps

    Parameters
    ----------
    ophys_timestamps : np.array
    df : pd.dataframe
        ith columns:
            timestamps (timestamps of signal)
            values  (signal of interest)

    Returns
    -------
    pd.dataFrame
        timestamps 
        values (values interpolated onto timestamps)
    """
    f = scipy.interpolate.interp1d(
        df['timestamps'],
        df['values'],
        bounds_error=False
    )

    interpolated = pd.DataFrame({
        'timestamps':ophys_timestamps,
        'values':f(ophys_timestamps)
    })

    return interpolated


def get_rewards(bod, ophys_timestamps):
    ''' Get rewards, seperated by auto_rewards,
    matched to ophys_timestamps (binary)
    '''
    rewards = bod.rewards.copy(deep=True)
    auto_rewards_pre = rewards.query('auto_rewarded == True')
    auto_rewards_bin = np.zeros(len(ophys_timestamps), 'bool')
    auto_rewards_bin[np.searchsorted(ophys_timestamps, auto_rewards_pre.timestamps)] = True
    rewards_pre = rewards.query('auto_rewarded == False')
    rewards_bin = np.zeros(len(ophys_timestamps), 'bool')
    rewards_bin[np.searchsorted(ophys_timestamps, rewards_pre.timestamps)] = True
    return auto_rewards_bin, rewards_bin