import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import seaborn as sns
import matplotlib.pyplot as plt

import multiprocessing as mp
from brain_observatory_utilities.datasets.optical_physiology.data_formatting import get_stimulus_response_df


# TODO: change to brain_observatory utilities
# from mindscope_utilities.visual_behavior_ophys.data_formatting import get_stimulus_response_df

# once group is here
#from .experiment_group import ExperimentGroup


# def get_mean_stimulus_response_expt_group(expt_group: ExperimentGroup,
#                                           event_type: str = "changes",
#                                           load_from_file: bool = True) -> pd.DataFrame:
#     """
#     Get mean stimulus response for each cell in the experiment group

#     Parameters
#     ----------
#     expt_grp: ExperimentGroup
#         An ExperimentGroup object
#     event_type: str
#         The event type ["changes", "omissions", "images", "all"]
#         see get_stimulus_response_df() for more details

#     Returns
#     -------
#     pd.DataFrame
#         A dataframe with the mean response traces & metrics
#         trace metrics = mean_trace, sem_trace, trace_timstamps
#         response metrics = mean_response, sem_response, mean_baseline,
#                             sem_baseline, response_latency, fano_factor,
#                             peak_response, time_to_peak, reliablity

#                             p_value, respone_window_duration,
#                             sd_over_baseline,
#                             fraction_significant_p_value_gray_screen
#                             correlation_values
#         """

#     mdfs = []
#     for expt_id, expt in expt_group.experiments.items():
#         try:
#             sdf = _get_stimulus_response_df(expt, event_type=event_type,
#                                             load_from_file=load_from_file,
#                                             save_to_file=False)

#             mdf = get_standard_mean_df(sdf)
#             mdf["ophys_experiment_id"] = expt_id
#             mdf["event_type"] = event_type
#             mdfs.append(mdf)
#         except Exception as e:
#             print(f"Failed to get stim response for: {expt_id}, {e}")

#     mdfs = pd.concat(mdfs).reset_index(drop=True)

#     # cells_filtered has expt_table info that is useful
#     expt_table = expt_group.expt_table
#     oct = expt_group.grp_ophys_cells_table.reset_index()

#     # calculate more metrics, will likely move to own functions
#     mdfs["mean_baseline_diff"] = mdfs["mean_response"] - \
#         mdfs["mean_baseline"]
#     mdfs["mean_baseline_diff_trace"] = mdfs["mean_trace"] - \
#         mdfs["mean_baseline"]

#     merged_mdfs = (mdfs.merge(expt_table, on=["ophys_experiment_id"])
#                        .merge(oct, on=["cell_roi_id"]))

#     return merged_mdfs


# def _get_stimulus_response_df(experiment: Union[BehaviorOphysExperiment, BehaviorOphysExperimentDev],
#                               event_type: str = 'changes',
#                               output_sampling_rate: float = 10.7,
#                               save_to_file: bool = False,
#                               load_from_file: bool = False,
#                               cache_dir: Union[str, Path] = "/allen/programs/mindscope/workgroups/learning/qc_plots/dev/mattd/3_lamf_mice/stim_response_cache"):
#     """Helper function for get_stimulus_response_df

#     Parameters
#     ----------
#     experiment: BehaviorOphysExperiment or BehaviorOphysExperimentDev
#         An experiment object
#     event_type: str
#         The event type ["changes", "omissions", "images", "all"]
#         see get_stimulus_response_df() for more details
#     output_sampling_rate: float
#         The sampling rate of the output trace
#     save_to_file: bool
#         If True, save the stimulus response dataframe to a file
#     load_from_file: bool
#         If True, load the stimulus response dataframe from a file

#     Returns
#     -------
#     pd.DataFrame
#         A dataframe with the stimulus response traces & metrics

#     # TODO: unhardcode cache_dir
#     # TODO: output_sampling_rate smart calculation

#     """
#     if not isinstance(experiment, (BehaviorOphysExperiment, BehaviorOphysExperimentDev)):
#         raise TypeError("experiment must be a BehaviorOphysExperiment or BehaviorOphysExperimentDev")

#     if event_type not in ["changes", "omissions", "images", "all"]:
#         raise ValueError("event_type must be one of ['changes', 'omissions', 'images', 'all']")

#     if save_to_file and load_from_file:
#         raise ValueError("save_to_file and load_from_file cannot both be True")

#     expt_id = experiment.metadata["ophys_experiment_id"]

#     cache_dir = Path(cache_dir)

#     # dev object can report correct frame rate, but different frame
#     # rate are possible across sessions, this would produce different
#     # trace sizes in stim response df, do for now just use constant 10.7.
#     # consider intelligent ways of handling this
#     # frame_rate = experiment.metadata["ophys_frame_rate"]

#     try:
#         fn = f"{expt_id}_{event_type}.pkl"
#         if (cache_dir / fn).exists() and load_from_file:
#             sdf = pd.read_pickle(cache_dir / fn)
#             print(f"Loading stim response df for {expt_id} from file")
#         else:
#             sdf = get_stimulus_response_df(experiment,
#                                            event_type=event_type,
#                                            output_sampling_rate=output_sampling_rate)
#             if save_to_file:
#                 if (cache_dir / fn).exists():
#                     print(f"Overwriting stim response df for {expt_id} in file")
#                 sdf.to_pickle(cache_dir / fn)
#                 print(f"Saving stim response df for {expt_id} to file")
#         return sdf
#     except Exception as e:
#         print(f"Failed to get stim response for: {expt_id}, {e}")
#         return None


####################################################################################################
# utilites
####################################################################################################

# # TODO: clean + document OLD
# def get_standard_mean_df(sr_df):
#     time_window = [-3, 3.1]
#     get_pref_stim = False  # relevant to image_name conditions

#     if "response_window_duration" in sr_df.keys():
#         response_window_duration = sr_df.response_window_duration.values[0]

#     output_sampling_rate = sr_df.ophys_frame_rate.unique()[0]
#     conditions = ["cell_roi_id"]
#     msr_df = get_mean_df(sr_df, 
#                          conditions=conditions,
#                          frame_rate=output_sampling_rate,
#                          window_around_timepoint_seconds=time_window,
#                          response_window_duration_seconds=response_window_duration,
#                          get_pref_stim=get_pref_stim,
#                          exclude_omitted_from_pref_stim=True)
#     return msr_df


# TODO: clean + document
def mean_stim_response_df(stim_response_df: pd.DataFrame,
                          conditions=["cell_roi_id"],
                          frame_rate=None,
                          window_around_timepoint_seconds: list = [-3, 3],
                          response_window_duration_seconds: float = None,
                          get_pref_stim=False,
                          exclude_omitted_from_pref_stim=True):
    """
    # MJD NOTES

    1) groupby "conditions": "cell" makes sense, TODO: "change_image_name" tho?
    2) apply get_mean_sem_trace()
    3) "response_window_duration_seconds" in df already
    4) "frame_rate" in df already
    5) get_pre_stim:
    """

    if frame_rate is None:
        try:
            frame_rate = stim_response_df.ophys_frame_rate.unique()[0]
        except Exception as e:
            print("Frame rate not found in stim_response_df, please provide frame_rate as an argument")

    if response_window_duration_seconds is None:
        try:
            response_window_duration = stim_response_df.response_window_duration.values[0]
        except Exception as e:
            response_window_duration = 0.5
            print("Response window duration not found in stim_response_df, using default value of 0.5 seconds")

    window = window_around_timepoint_seconds

    rdf = stim_response_df.copy()


    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace',
               'trace_timestamps', 'mean_responses', 'mean_baseline', 'sem_baseline']]
    mdf = mdf.reset_index()

    # save response window duration as a column for reference
    mdf['response_window_duration'] = response_window_duration

    if get_pref_stim:
        if ('image_name' in conditions) or ('change_image_name' in conditions) or ('prior_image_name' in conditions):
            mdf = annotate_mean_df_with_pref_stim(
                mdf, exclude_omitted_from_pref_stim)

    try:
        mdf = annotate_mean_df_with_fano_factor(mdf)
        mdf = annotate_mean_df_with_time_to_peak(mdf, window, frame_rate)
        mdf = annotate_mean_df_with_p_value(
            mdf, window, response_window_duration, frame_rate)
        mdf = annotate_mean_df_with_sd_over_baseline(
            mdf, window, response_window_duration, frame_rate)
    except Exception as e:  # NOQA E722
        print(e)
        pass

    if 'p_value_gray_screen' in rdf.keys():
        fraction_significant_p_value_gray_screen = rdf.groupby(conditions).apply(
            get_fraction_significant_p_value_gray_screen)
        fraction_significant_p_value_gray_screen = fraction_significant_p_value_gray_screen.reset_index()
        mdf['fraction_significant_p_value_gray_screen'] = fraction_significant_p_value_gray_screen.fraction_significant_p_value_gray_screen

    try:
        reliability = rdf.groupby(conditions).apply(
            compute_reliability, window, response_window_duration, frame_rate)
        reliability = reliability.reset_index()
        mdf['reliability'] = reliability.reliability
        mdf['correlation_values'] = reliability.correlation_values
        # print('done computing reliability')
    except Exception as e:
        print('failed to compute reliability')
        print(e)
        pass

    mdf["mean_baseline_diff"] = mdf["mean_response"] - mdf["mean_baseline"]
    mdf["mean_baseline_diff_trace"] = mdf["mean_trace"] - mdf["mean_baseline"]

    if 'index' in mdf.keys():
        mdf = mdf.drop(columns=['index'])
    return mdf


####################################################################################################
# trace metrics
####################################################################################################


def get_successive_frame_list(timepoints_array, timestanps):
    # This is a modification of get_nearest_frame for speedup
    #  This implementation looks for the first 2p frame consecutive to the stim
    successive_frames = np.searchsorted(timestanps, timepoints_array)

    return successive_frames


def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    #   frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    frame_for_timepoint = get_successive_frame_list(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints


def get_responses_around_event_times(trace, timestamps, event_times, frame_rate, window=[-2, 3]):
    responses = []
    for event_time in event_times:
        response, times = get_trace_around_timepoint(event_time, trace, timestamps,
                                                     frame_rate=frame_rate, window=window)
        responses.append(response)
    responses = np.asarray(responses)
    return responses


def get_mean_in_window(trace, window, frame_rate, use_events=False):
    mean = np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])
    return mean


def get_sd_in_window(trace, window, frame_rate):
    std = np.std(
        trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])
    return std


def get_n_nonzero_in_window(trace, window, frame_rate):
    datapoints = trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))]
    n_nonzero = len(np.where(datapoints > 0)[0])
    return n_nonzero


def get_sd_over_baseline(trace, response_window, baseline_window, frame_rate):
    baseline_std = get_sd_in_window(trace, baseline_window, frame_rate)
    response_mean = get_mean_in_window(trace, response_window, frame_rate)
    return response_mean / (baseline_std)


def get_p_val(trace, response_window, frame_rate):
    from scipy import stats
    response_window_duration = response_window[1] - response_window[0]
    baseline_end = int(response_window[0] * frame_rate)
    baseline_start = int((response_window[0] - response_window_duration) * frame_rate)
    stim_start = int(response_window[0] * frame_rate)
    stim_end = int((response_window[0] + response_window_duration) * frame_rate)
    (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
    return p

####################################################################################################
# metrics for grouped stim_response_df
####################################################################################################


# TODO: clean + document
def get_mean_sem_trace(group):
    mean_response = np.mean(group['mean_response'])
    mean_baseline = np.mean(group['baseline_response'])
    mean_responses = group['mean_response'].values
    sem_response = np.std(group['mean_response'].values) / \
        np.sqrt(len(group['mean_response'].values))
    sem_baseline = np.std(group['baseline_response'].values) / \
        np.sqrt(len(group['baseline_response'].values))
    mean_trace = np.mean(group['trace'], axis=0)
    sem_trace = np.std(group['trace'].values) / \
        np.sqrt(len(group['trace'].values))
    trace_timestamps = np.mean(group['trace_timestamps'], axis=0)
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_baseline': mean_baseline, 'sem_baseline': sem_baseline,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace,
                      'trace_timestamps': trace_timestamps,
                      'mean_responses': mean_responses})


# TODO: clean + document
def get_mean_sem(group):
    mean_response = np.mean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / \
        np.sqrt(len(group['mean_response'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response})


# TODO: clean + document
def get_fraction_significant_trials(group):
    fraction_significant_trials = len(
        group[group.p_value < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_trials': fraction_significant_trials})


# TODO: clean + document
def get_fraction_significant_p_value_gray_screen(group):
    fraction_significant_p_value_gray_screen = len(
        group[group.p_value_gray_screen < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_gray_screen': fraction_significant_p_value_gray_screen})


# TODO: clean + document
def get_fraction_significant_p_value_omission(group):
    fraction_significant_p_value_omission = len(
        group[group.p_value_omission < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_omission': fraction_significant_p_value_omission})


# TODO: clean + document
def get_fraction_significant_p_value_stimulus(group):
    fraction_significant_p_value_stimulus = len(
        group[group.p_value_stimulus < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_stimulus': fraction_significant_p_value_stimulus})


# TODO: clean + document
def get_fraction_active_trials(group):
    fraction_active_trials = len(
        group[group.mean_response > 0.05]) / float(len(group))
    return pd.Series({'fraction_active_trials': fraction_active_trials})


# TODO: clean + document
def get_fraction_responsive_trials(group):
    fraction_responsive_trials = len(
        group[(group.p_value_baseline < 0.05)]) / float(len(group))
    return pd.Series({'fraction_responsive_trials': fraction_responsive_trials})


# TODO: clean + document
def get_fraction_nonzero_trials(group):
    fraction_nonzero_trials = len(
        group[group.n_events > 0]) / float(len(group))
    return pd.Series({'fraction_nonzero_trials': fraction_nonzero_trials})


# TODO: clean + document
def compute_reliability_vectorized(traces):
    '''
    Compute average pearson correlation between pairs of rows of the input matrix.
    Args:
        traces(np.ndarray): trace array with shape m*n, with m traces and n trace timepoints
    Returns:
        reliability (float): Average correlation between pairs of rows
    '''
    # Compute m*m pearson product moment correlation matrix between rows of input.
    # This matrix is 1 on the diagonal (correlation with self) and mirrored across
    # the diagonal (corr(A, B) = corr(B, A))
    corrmat = np.corrcoef(traces)
    # We want the inds of the lower triangle, without the diagonal, to average
    m = traces.shape[0]
    lower_tri_inds = np.where(np.tril(np.ones([m, m]), k=-1))
    # Take the lower triangle values from the corrmat and averge them
    correlation_values = list(corrmat[lower_tri_inds[0], lower_tri_inds[1]])
    reliability = np.nanmean(correlation_values)
    return reliability, correlation_values


# TODO: clean + document
def compute_reliability(group, window=[-3, 3], response_window_duration=0.5, frame_rate=30.):
    # computes trial to trial correlation across input traces in group,
    # only for portion of the trace after the change time or flash onset time

    onset = int(np.abs(window[0]) * frame_rate)
    response_window = [onset, onset + (int(response_window_duration * frame_rate))]
    traces = group['trace'].values
    traces = np.vstack(traces)
    if traces.shape[0] > 5:
        # limit to response window
        traces = traces[:, response_window[0]:response_window[1]]
        reliability, correlation_values = compute_reliability_vectorized(
            traces)
    else:
        reliability = np.nan
        correlation_values = []
    return pd.Series({'reliability': reliability, 'correlation_values': correlation_values})


####################################################################################################
# Annotate various data frames
####################################################################################################


# TODO: clean + document
def get_time_to_peak(trace, window=[-4, 8], frame_rate=30.):
    response_window_duration = 0.75
    response_window = [np.abs(window[0]), np.abs(
        window[0]) + response_window_duration]
    response_window_trace = trace[int(
        response_window[0] * frame_rate):(int(response_window[1] * frame_rate))]
    peak_response = np.amax(response_window_trace)
    peak_frames_from_response_window_start = np.where(
        response_window_trace == np.amax(response_window_trace))[0][0]
    time_to_peak = peak_frames_from_response_window_start / float(frame_rate)
    return peak_response, time_to_peak


# TODO: clean + document
def annotate_mean_df_with_time_to_peak(mean_df, window=[-4, 8], frame_rate=30.):
    ttp_list = []
    peak_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        peak_response, time_to_peak = get_time_to_peak(
            mean_trace, window=window, frame_rate=frame_rate)
        ttp_list.append(time_to_peak)
        peak_list.append(peak_response)
    mean_df['peak_response'] = peak_list
    mean_df['time_to_peak'] = ttp_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_fano_factor(mean_df):
    ff_list = []
    for idx in mean_df.index:
        mean_responses = mean_df.iloc[idx].mean_responses
        sd = np.nanstd(mean_responses)
        mean_response = np.nanmean(mean_responses)
        # take abs value to account for negative mean_response
        fano_factor = np.abs((sd * 2) / mean_response)
        ff_list.append(fano_factor)
    mean_df['fano_factor'] = ff_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_p_value(mean_df, window=[-4, 8], response_window_duration=0.5, frame_rate=30.):
    response_window = [np.abs(window[0]), np.abs(
        window[0]) + response_window_duration]
    p_val_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        p_value = get_p_val(mean_trace, response_window, frame_rate)
        p_val_list.append(p_value)
    mean_df['p_value'] = p_val_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_sd_over_baseline(mean_df, window=[-4, 8], response_window_duration=0.5, frame_rate=30.):
    response_window = [np.abs(window[0]), np.abs(
        window[0]) + response_window_duration]
    baseline_window = [
        np.abs(window[0]) - response_window_duration, (np.abs(window[0]))]
    sd_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        sd = get_sd_over_baseline(
            mean_trace, response_window, baseline_window, frame_rate)
        sd_list.append(sd)
    mean_df['sd_over_baseline'] = sd_list
    return mean_df


# TODO: clean + document
def annotate_mean_df_with_pref_stim(mean_df, exclude_omitted_from_pref_stim=True):
    if 'prior_image_name' in mean_df.keys():
        image_name = 'prior_image_name'
    elif 'image_name' in mean_df.keys():
        image_name = 'image_name'
    else:
        image_name = 'change_image_name'
    mdf = mean_df.reset_index()
    mdf['pref_stim'] = False
    if 'cell_specimen_id' in mdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    for cell in mdf[cell_key].unique():
        mc = mdf[(mdf[cell_key] == cell)]
        if exclude_omitted_from_pref_stim:
            if 'omitted' in mdf[image_name].unique():
                mc = mc[mc[image_name] != 'omitted']
        pref_image = mc[(mc.mean_response == np.max(
            mc.mean_response.values))][image_name].values[0]
        row = mdf[(mdf[cell_key] == cell) & (
            mdf[image_name] == pref_image)].index
        mdf.loc[row, 'pref_stim'] = True
    return mdf


# TODO: clean + document
def annotate_trial_response_df_with_pref_stim(trial_response_df):
    rdf = trial_response_df.copy()
    rdf['pref_stim'] = False
    if 'cell_specimen_id' in rdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    mean_response = rdf.groupby(
        [cell_key, 'change_image_name']).apply(get_mean_sem_trace)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = np.where(m.loc[cell]['mean_response'].values == np.max(
            m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = rdf[(rdf[cell_key] == cell) & (
            rdf.change_image_name == pref_image)].index
        for trial in trials:
            rdf.loc[trial, 'pref_stim'] = True
    return rdf


# TODO: clean + document
def annotate_flash_response_df_with_pref_stim(fdf):
    fdf = fdf.reset_index()
    if 'cell_specimen_id' in fdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    fdf['pref_stim'] = False
    mean_response = fdf.groupby([cell_key, 'image_name']).apply(get_mean_sem)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = \
            np.where(m.loc[cell]['mean_response'].values == np.nanmax(
                m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = fdf[(fdf[cell_key] == cell) & (
            fdf.image_name == pref_image)].index
        for trial in trials:
            fdf.loc[trial, 'pref_stim'] = True
    return fdf


# TODO: clean + document
def annotate_flashes_with_reward_rate(dataset):
    last_time = 0
    reward_rate_by_frame = []
    trials = dataset.trials[dataset.trials.trial_type != 'aborted']
    flashes = dataset.stimulus_table.copy()
    for change_time in trials.change_time.values:
        reward_rate = trials[trials.change_time == change_time].reward_rate.values[0]
        for start_time in flashes.start_time:
            if (start_time < change_time) and (start_time > last_time):
                reward_rate_by_frame.append(reward_rate)
                last_time = start_time
    # fill the last flashes with last value
    for i in range(len(flashes) - len(reward_rate_by_frame)):
        reward_rate_by_frame.append(reward_rate_by_frame[-1])
    flashes['reward_rate'] = reward_rate_by_frame
    return flashes


######################################################################
# multiprocessing
#######################################################################

def _process_stim_response(dataset, data_type, event_type):
        # Call your function with specific data_type and event_type
        return (event_type, data_type, get_stimulus_response_df(dataset,
                                                                data_type=data_type,
                                                                event_type=event_type,
                                                                time_window=[-3, 3],
                                                                interpolate=False,
                                                                output_sampling_rate=None,
                                                                response_window_duration=0.5))

def stim_response_all_mp(dataset):

    data_types = ['dff', 'events']
    event_types = ["changes", "images", "omissions"]

    # Create a multiprocessing pool
    pool = mp.Pool()

    results_dict = {}

    # Map the function across all combinations of data_types and event_types
    for event_type in event_types:
        for data_type in data_types:
            result = pool.apply_async(_process_stim_response, (dataset, data_type, event_type))
            results_dict[(event_type, data_type)] = result


    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Process the results from the dictionary
    for key, result in results_dict.items():
        event_type, data_type = key
        _, _, sr_df = result.get()  # Retrieve data_type, event_type, and sr_df from result
        print(f"Result for event_type={event_type}, data_type={data_type}:")
        results_dict[key] = sr_df

    return results_dict

def stim_response_all(dataset):
    """no multiprocessing, just for loop"""

    data_types = ['dff', 'events']
    event_types = ["changes", "images", "omissions"]

    results_dict = {}

    # Map the function across all combinations of data_types and event_types
    for event_type in event_types:
        for data_type in data_types:


            if event_type in ['images', 'changes']:
                response_window_duration = 0.5
                time_window = [-0.5, 0.75]
            elif event_type is 'omissions':
                response_window_duration = 0.75
                time_window = [-3, 3]


            print(f"Processing event_type={event_type}, data_type={data_type}")
            results_dict[(event_type, data_type)] = get_stimulus_response_df(dataset,
                                                                data_type=data_type,
                                                                event_type=event_type,
                                                                time_window=time_window,
                                                                interpolate=False,
                                                                output_sampling_rate=None,
                                                                response_window_duration=response_window_duration)

    return results_dict

######################################################################
# Plotting: stim response 
######################################################################



def plot_stim_response_for_roi(sr_df, cell_roi_id, data_type, event_type):
    """Plot the stimulus repsonse for a specific cell_roi_id
    """
    cell_df = sr_df[sr_df.cell_roi_id == cell_roi_id]
    traces = np.vstack(cell_df.trace.values)
    mean_trace = np.nanmean(traces, axis=0)
    timestamps = np.round(cell_df['trace_timestamps'].values[0],2)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(timestamps, mean_trace)
    plt.xlabel('2P frames')
    plt.ylabel(f'{data_type}')
    plt.title(f'Mean population response to {event_type}')
    
    plt.axvline(0, color='r', linestyle='--')
    plt.show()

def plot_stim_response_population_mean(sr_df, data_type, event_type, ax=None, title=None):
    """
    Plot the mean population stimulus response
    """
    sns.set_style('darkgrid')
    sns.set_context('talk')

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    if title is None:
        ax.set_title(f'Mean population response to {event_type}')
    else:
        ax.set_title(title)

    

    # processes data
    traces = np.vstack(sr_df.trace.values)
    mean_trace = np.nanmean(traces, axis=0)
    timestamps = np.round(sr_df['trace_timestamps'].values[0],2)
    
    ax.plot(timestamps, mean_trace)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{data_type}')
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='stimulus onset')

    # actually keeps legend on last plot
    plt.legend(loc='upper left', fontsize='small', frameon=False)

###########
# Plotting: mean stim response
########

def plot_mean_stim_response_heatmap(msr_df,
                                    event_type,
                                    data_type, 
                                    trace_type='mean_trace',
                                    ax=None):


    sns.set_style("white")

    if ax is None:
        fig, ax = plt.subplots(1,figsize=(10, 20))

    mean_traces = np.vstack(msr_df[trace_type].values)
    timestamps = np.round(msr_df['trace_timestamps'].values[0],2)

    vmax = np.percentile(mean_traces, 99.7)
    vmin = np.percentile(mean_traces, 5.0)

    ax.imshow(mean_traces, aspect='auto',vmax=vmax, vmin=vmin)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cell index')

    
    cbar = plt.colorbar(ax.imshow(mean_traces, aspect='auto', vmax=vmax, vmin=vmin), ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label('Events')

    return ax



def plot_mean_stim_response_for_roi(msr_df, cell_roi_id, data_type, event_type,ax=None):
    """
    Plot the stimulus response for a specific cell_roi_id
    """
    cell_df = msr_df[msr_df.cell_roi_id == cell_roi_id]
    timestamps = np.round(cell_df['trace_timestamps'].values[0],2)
    mean_trace = cell_df['mean_trace'].values[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(timestamps, mean_trace)
    ax.set_xlabel('2P frames')
    ax.set_ylabel(f'{data_type}')
    ax.set_title(f'Mean population response to {event_type}')
    
    ax.axvline(0, color='r', linestyle='--')
    plt.show()


def plot_mean_stim_response_top_ten(msr_df, data_type, event_type, response_trace="mean_trace"):
    """
    Plot the mean response of the top 10 cells

    Parameters
    ----------
    msr_df : pd.DataFrame
        Mean stimulus response DataFrame
    data_type : str
        Data type of the response
    event_type : str
        Type of event
    response_trace : str
        Column name of the response
        cols = ['mean_trace', 'mean_baseline_diff_trace]

    Returns
    -------
    None
    """

    y_scale = 0.05
    mean_traces = np.vstack(msr_df[response_trace].values)
    timestamps = np.round(msr_df['trace_timestamps'].values[0],2)

    plt.figure(figsize=(5, 10))
    sns.set_style('darkgrid')
    for i in range(10):
        plt.plot(timestamps, mean_traces[i] + i*y_scale)

    # ylabel, mean response, top 10 cells
    plt.yticks(np.arange(10)*y_scale, msr_df['cell_roi_id'][:10])
    plt.ylabel(f'Mean response {data_type}')
    plt.xlabel('2P frames')
    plt.title(f'Top 10 cells by mean response to {event_type}')

#############################
# Figures
#############################


def fig_stim_response(results_dict):
    """
    """
    sns.set_style('darkgrid')
    sns.set_context('talk')

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, (event_type, data_type) in enumerate(results_dict.keys()):
        sr_df = results_dict[(event_type, data_type)]
        plot_stim_response_population_mean(sr_df, data_type, event_type, 
                                              ax=axes[i], title=f"{event_type}")
    plt.tight_layout()
    plt.show()