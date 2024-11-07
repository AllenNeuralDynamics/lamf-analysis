import numpy as np
import h5py
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_dff_h5(dff_file, remove_nan_rows=True):
    with h5py.File(dff_file, "r") as f:
        dff = f["data"][:]
        roi_names = f["roi_names"][:]

    if remove_nan_rows:
        # if row all nans, remove from dff and roi_names
        nan_rows = np.isnan(dff).all(axis=1)
        dff = dff[~nan_rows]
        roi_names = roi_names[~nan_rows]

    # sort top 1 percentile of dff values by mean
    top_1_percentile = np.percentile(dff, 99, axis=1)
    dff_mean = np.mean(top_1_percentile)
    # descending order
    sort_idx = np.argsort(top_1_percentile, axis=0)[::-1]
    dff = dff[sort_idx]
    roi_names = roi_names[sort_idx]

    return dff,roi_names


# def dff_robust_noise(dff_trace):
#     """Robust estimate of std of noise in df/f

#     Arguments:
#         dff_trace {[type]} -- [description]

#     Returns:
#         [type] -- [description]
#     """

#     sigma_MAD_conversion_factor = 1.4826

#     dff_trace = np.asarray(dff_trace)
#     # first pass removing big pos peaks
#     dff_trace = dff_trace[dff_trace < 1.5 * np.abs(dff_trace.min())]
#     MAD = np.median(np.abs(dff_trace - np.median(dff_trace)))  # MAD = median absolute deviation
#     robust_standard_deviation = sigma_MAD_conversion_factor * MAD

#     # second pass removing remaining pos and neg peaks
#     dff_trace = dff_trace[np.abs(dff_trace - np.median(dff_trace)) < 2.5 * robust_standard_deviation]
#     MAD = np.median(np.abs(dff_trace - np.median(dff_trace)))
#     robust_standard_deviation = sigma_MAD_conversion_factor * MAD
#     return robust_standard_deviation

# def dff_robust_signal(dff_trace, robust_standard_deviation):
#     """ median deviation

#     Arguments:
#         dff_trace {[type]} -- [description]
#         robust_standard_deviation {[type]} -- [description]

#     Returns:
#         [type] -- [description]
#     """
#     dff_trace = np.asarray(dff_trace)
#     median_deviation = np.median(dff_trace[(dff_trace - np.median(dff_trace)) > robust_standard_deviation])
#     return median_deviation

# def dff_robust_snr(robust_signal, robust_noise):
#     mean_snr = np.mean(robust_signal / robust_noise)
#     return

# def compute_robust_snr_on_dataframe(dataframe):
#     """takes a dataframe with a "dff" column that has the dff trace array
#         for a cell_specimen_id and for noise uses Robust estimate of std for signal
#         uses median deviation, and for robust snr the robust signal / robust noise

#     Arguments:
#         dataframe {[type]} -- [description]

#     Returns:
#         dataframe -- input dataframe but with the following columns added:
#                         "robust_noise"
#                         "robust_signal"
#                         "robust_snr"
#     """
#     dataframe = dataframe.copy()
#     if 'dff' in dataframe.columns:
#         column = 'dff'
#     elif 'filtered_events' in dataframe.columns:
#         column = 'filtered_events'
#     dataframe['robust_noise'] = dataframe.apply(lambda x: dff_robust_noise(x[column]), axis=1) 
#     dataframe["robust_signal"] = dataframe.apply(lambda x: dff_robust_signal(x[column], x["robust_noise"]), axis=1 )
#     dataframe['robust_snr']  = dataframe['robust_signal'] / dataframe['robust_noise']
#     return dataframe


# def top_percentile(dataframe, percentile=99,column="dff"):
    
#     def _top_percentile(dff_trace):
#         percentile_value = np.percentile(dff_trace, percentile)
#         return percentile_value
#     dataframe['top_1_percent'] = dataframe.copy().apply(lambda x: _top_percentile(x[column]), axis=1)
    
#     return dataframe

# def add_skewness_to_df(df,column="dff"):

#     def _skewness_row(trace):
#         return stats.skew(trace)

#     df['skewness'] = df.copy().apply(lambda x: _skewness_row(x[column]), axis=1)

#     return df
        
        
# def annotate_dff_metrics(dff_traces):
#     n_og_dff_traces = len(dff_traces)

#     dff_traces = add_skewness_to_df(dff_traces)

#     # remove rows in dff_traces where skewness is nan
#     dff_traces = dff_traces[dff_traces.skewness.notna()]
#     n_dff_traces = len(dff_traces)
#     print(f"Removed {n_og_dff_traces - n_dff_traces} rows from dff_traces where skewness was nan")
#     print(f"N dff_traces: {n_dff_traces}")

#     dff_traces = compute_robust_snr_on_dataframe(dff_traces)
    
#     dff_traces = top_percentile(dff_traces)

#     return dff_traces


####
# array
####


def dff_robust_noise(dff_trace: np.array) -> float:
    """Robust estimate of std of noise in df/f

    Parameters
    ----------
    dff_trace : np.array

    Returns
    -------
    float
    """

    sigma_MAD_conversion_factor = 1.4826
    # first pass removing big pos peaks
    dff_trace = dff_trace[dff_trace < 1.5 * np.abs(dff_trace.min())]
    MAD = np.median(np.abs(dff_trace - np.median(dff_trace)))  # MAD = median absolute deviation
    robust_standard_deviation = sigma_MAD_conversion_factor * MAD

    # second pass removing remaining pos and neg peaks
    dff_trace = dff_trace[np.abs(dff_trace - np.median(dff_trace)) < 2.5 * robust_standard_deviation]
    MAD = np.median(np.abs(dff_trace - np.median(dff_trace)))
    robust_standard_deviation = sigma_MAD_conversion_factor * MAD
    return robust_standard_deviation

def dff_robust_signal(dff_trace: np.array, 
                      robust_standard_deviation: float) -> float:
    """ median deviation

    Arguments:
        dff_trace {[type]} -- [description]
        robust_standard_deviation {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    dff_trace = np.asarray(dff_trace)
    median_deviation = np.median(dff_trace[(dff_trace - np.median(dff_trace)) > robust_standard_deviation])
    return median_deviation

def dff_robust_snr(robust_signal, robust_noise):
    mean_snr = np.mean(robust_signal / robust_noise)
    return

def compute_robust_snr_on_dataframe(dataframe):
    """takes a dataframe with a "dff" column that has the dff trace array
        for a cell_specimen_id and for noise uses Robust estimate of std for signal
        uses median deviation, and for robust snr the robust signal / robust noise

    Arguments:
        dataframe {[type]} -- [description]

    Returns:
        dataframe -- input dataframe but with the following columns added:
                        "robust_noise"
                        "robust_signal"
                        "robust_snr"
    """
    dataframe = dataframe.copy()
    if 'dff' in dataframe.columns:
        column = 'dff'
    elif 'filtered_events' in dataframe.columns:
        column = 'filtered_events'
    dataframe['robust_noise'] = dataframe.apply(lambda x: dff_robust_noise(x[column]), axis=1) 
    dataframe["robust_signal"] = dataframe.apply(lambda x: dff_robust_signal(x[column], x["robust_noise"]), axis=1 )
    dataframe['robust_snr']  = dataframe['robust_signal'] / dataframe['robust_noise']
    return dataframe

def top_percentile(dff_trace: np.array, percentile=99):
        percentile_value = np.percentile(dff_trace, percentile)
        return percentile_value

def skewness(dff_trace: np.array) -> float:
    return stats.skew(dff_trace)


def calc_dff_metrics(dff_traces: np.array, roi_ids: list=None) -> dict:
    """Calculate metrics for dff traces array (n_cells, n_frames)
    """

    metrics = {}

    if roi_ids is None:
        roi_ids = np.arange(dff_traces.shape[0])


    for id, trace in zip(roi_ids, dff_traces):
        metrics[id] = {}
        metrics[id]["skewness"] = skewness(trace)
        metrics[id]["top_percentile"] = top_percentile(trace)
        metrics[id]["robust_noise"] = dff_robust_noise(trace)
        metrics[id]["robust_signal"] = dff_robust_signal(trace, metrics[id]["robust_noise"])
        metrics[id]["robust_snr"] = metrics[id]["robust_signal"] / metrics[id]["robust_noise"]

    return metrics




        


####################################################################################################
# Plotting
####################################################################################################

def plot_dff_traces_examples(dff_traces, frame_rate = None, save=True, output_folder=None):
    """Plot top 10 skew traces, and zoom on most active 1000 frames

    """
    sns.set_style("white")
    sns.set_context("talk")
    #frame_rate = 42 # hz
    # drop rows with all nans
    dff_traces = dff_traces[~np.isnan(dff_traces).all(axis=1)]
    # calc skew for each trace
    skew = stats.skew(dff_traces, axis=1, nan_policy='omit')

    # get get -5 tp -10 skew
    top_skew_idx = np.argsort(skew)[-10:-5]
    top_skew = skew[top_skew_idx]

    # grid spec, 10x2, left column is trace, right column is zoom on most active 1000 frames
    # right columsn is 1/3 width of left column
    # plot each trace on diff axis

    fig = plt.figure(figsize=(12, 17))
    gs = fig.add_gridspec(10, 2)


    #for each row in top skew, find index of trace with largest trapz integral
    active_indices = []
    zoom_dff_epochs = []
    for trace in dff_traces[top_skew_idx]:
        n_chunks = len(trace) / 500
        epochs = np.array_split(trace, n_chunks)
        max_auc_idx = np.argmax([np.trapz(epoch) for epoch in epochs])
        active_indices.append(max_auc_idx)

        active_dff = epochs[max_auc_idx]
        active_dff = active_dff[:500]
        zoom_dff_epochs.append(active_dff)

    # # select last 500 frames of each trace
    # zoom_dff_epochs = []
    # for trace in dff_traces[top_skew_idx]:
    #     zoom_dff_epochs.append(trace[-500:])
    

    for i, idx in enumerate(top_skew_idx):
        skew = np.round(top_skew[i], 2)
        # plot trace
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(dff_traces[idx, :],linewidth=0.75)
        #ax.set_title(f"Skew: {skew}")

        # plot zoom
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(zoom_dff_epochs[i],linewidth=0.75)
        #ax.set_title(f"Skew: {skew}")

        # ax off top and right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if frame_rate:
            #convert x to seconds
            ax.set_xlabel("Time (s)")
            ax.set_xticks(np.arange(0, 2000, 500))
            ax.set_xticklabels(np.round(np.arange(0, 2000, 500) / frame_rate))



    plt.tight_layout()

import seaborn as sns
def plot_population_dff(dff_traces, vmin=None, vmax=None, title=None):
    sns.set_style("white")
    sns.set_context("poster")
    y_scale = dff_traces.shape[0] / 100 # 100 works for 80 cells
    y_scale = 1.4
    fig, ax = plt.subplots(1, 1, figsize=(25, 10*y_scale))

    if vmin is None:
        vmin = np.percentile(dff_traces, 15)
    if vmax is None:
        vmax = np.percentile(dff_traces, 98)
    plt.imshow(dff_traces, aspect='auto', cmap='viridis',vmin=vmin, vmax=vmax)

    if title is not None:
        plt.title(title)
    else:
        plt.title("Population dff traces")

    plt.xlabel("Time (frames)")
    plt.ylabel("Cell index")

    # ytick every 100 cells
    ax.set_yticks(np.arange(0, dff_traces.shape[0], 100))
    ax.set_yticklabels(np.arange(0, dff_traces.shape[0],
                                100).astype(int))

    # show colorbar label dff
    cbar = plt.colorbar()
    cbar.set_label("df/f")

    return fig, ax



def plot_population_dff_normalize(dff_traces, vmin=None, vmax=None, title=None):
    sns.set_context("poster")
    y_scale = dff_traces.shape[0] / 100 # 100 works for 80 cells
    y_scale = 1.2
    fig, ax = plt.subplots(1, 1, figsize=(25, 10*y_scale))
    
    # normalize each row to half max
    dff_traces = dff_traces / np.max(dff_traces, axis=1)[:, None]
    vmin = 0
    vmax = .4
    plt.imshow(dff_traces, aspect='auto', cmap='viridis',vmin=vmin, vmax=vmax)

    if title is not None:
        plt.title(title)
    else:
        plt.title("Population dff traces - normlize ")

    plt.xlabel("Time (frames)")
    plt.ylabel("Cell index")

    # ytick every 100 cells
    ax.set_yticks(np.arange(0, dff_traces.shape[0], 100))
    ax.set_yticklabels(np.arange(0, dff_traces.shape[0],
                                100).astype(int))

    # show colorbar label dff
    cbar = plt.colorbar()
    cbar.set_label("df/f (norm half max)")

    return fig, ax