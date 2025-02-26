import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

from lamf_analysis import utils as lamf_utils
from lamf_analysis.visualization import vba_tools
from lamf_analysis.code_ocean import capsule_data_utils as cdu
from lamf_analysis.code_ocean import capsule_bod_utils as cbu


################################################
## For plotting session QC plots

def get_session_traces(bod_list):
    ''' Get neuronal activity and behavior traces for one session
    Neuronal activities are from all planes in the session (in the bod_list)
    '''
    dff_traces = []
    event_traces = []
    num_cells_per_plane = []
    for bod in bod_list:
        valid_roi_ids = bod.cell_specimen_table.query('valid_roi').cell_roi_id.values
        dff_traces.append(np.vstack(bod.dff_traces.loc[valid_roi_ids].dff.values))
        event_traces.append(np.vstack(bod.events.loc[valid_roi_ids].events.values))
        num_cells_per_plane.append(len(valid_roi_ids))
    dff_traces = np.vstack(dff_traces)
    event_traces = np.vstack(event_traces)
    ophys_timestamps = bod.ophys_timestamps.values

    running_speed = cbu.get_running_speed(bod, ophys_timestamps)
    pupil = cbu.get_pupil_area(bod, ophys_timestamps)
    auto_rewards_mask, rewards_mask = cbu.get_rewards(bod, ophys_timestamps)

    return dff_traces, event_traces, num_cells_per_plane, \
        running_speed, pupil, auto_rewards_mask, rewards_mask, \
            ophys_timestamps


def plot_session_traces(dff_traces, event_traces, num_cells_per_plane,
                        running_speed, pupil, auto_rewards_mask, rewards_mask,
                        ophys_timestamps, session_row, timestamp_tick_interval=10,
                        dff_cmap='PRGn', event_vmax_percentile=80, event_vmax_adjust=1.0):
    ''' Plot neuronal activity and behavior traces for one session
    '''
    fig, ax = plt.subplots(6,1,figsize=(12,15), sharex=True, gridspec_kw={'height_ratios': [4,4,1,1,1,1]})

    # dFF raster
    vmin = np.percentile(dff_traces, 2)
    vmax = np.percentile(dff_traces, 95)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im_dff = ax[0].imshow(dff_traces, aspect='auto', cmap=dff_cmap, norm=norm)

    timestamps = ophys_timestamps - ophys_timestamps[0]
    timestamp_tick_interval = 10 # in minutes
    timestamp_tick_values = np.array([int(t) for t in np.arange(0, timestamps[-1]/60, 10)])
    timestamp_tick_inds = np.array([np.argmin(np.abs(timestamps / 60 - t)) for t in timestamp_tick_values])
    ax[0].set_xticks(timestamp_tick_inds)
    ax[0].set_xticklabels(timestamp_tick_values, fontsize=12)
    ax[0].set_ylabel('Cell #', fontsize=15)

    ax[0].set_yticks(np.cumsum(num_cells_per_plane))
    for num_cells in np.cumsum(num_cells_per_plane):
        ax[0].axhline(num_cells, color='k', linestyle='--', linewidth=1)
    ax[0].set_title('dF/F', fontsize=15, fontweight='bold', loc='left')

    # Events raster
    vmin = 0
    pos_values = event_traces[event_traces > 0]
    vmax = np.percentile(pos_values, event_vmax_percentile) * event_vmax_adjust
    im_events = ax[1].imshow(event_traces, aspect='auto', cmap='gist_yarg', vmin=vmin, vmax=vmax)

    ax[1].set_ylabel('Cell #', fontsize=15)
    ax[1].set_yticks(np.cumsum(num_cells_per_plane))
    for num_cells in np.cumsum(num_cells_per_plane):
        ax[1].axhline(num_cells, color='C1', linestyle='--', linewidth=1)
    ax[1].set_title('Events', fontsize=15, fontweight='bold', loc='left')

    dff_mean_trace = dff_traces.mean(axis=0)
    ax[2].plot(dff_mean_trace, color='g')
    ax[2].set_ylabel('Mean\ndF/F', fontsize=15)

    events_mean_trace = event_traces.mean(axis=0)
    ax[3].plot(events_mean_trace, color='k')
    ax[3].set_ylabel('Mean\nEvents', fontsize=15)

    ax[4].plot(running_speed.running_speed.values, color='C0')
    ax[4].set_ylabel('Running\n(cm/s)', fontsize=15)
    ax[5].plot(pupil.pupil_area.values, color='C1')
    ax[5].set_ylabel('Pupil\nArea (AU)', fontsize=15)
    ax[5].set_xlabel('Time (min)', fontsize=15)

    # add rewards
    # auto rewards as black, rewards as cyan (correct lick in OPHYS_6 as magenta)
    # at the top of pupil trace axis
    rewards_label = 'correct lick' if 'OPHYS_6' in session_row.session_type else 'rewards'
    rewards_color = 'm' if 'OPHYS_6' in session_row.session_type else 'c'
    max_pupil_area = pupil.pupil_area.max()
    ax[5].scatter(np.where(auto_rewards_mask)[0], 
                    [max_pupil_area] * len(np.where(auto_rewards_mask)[0]),
                    color='k', marker='|', label='auto rewards')
    ax[5].scatter(np.where(rewards_mask)[0], 
                    [max_pupil_area] * len(np.where(rewards_mask)[0]),
                    color=rewards_color, marker='|', label=rewards_label)
    ax[5].legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1.5), fontsize=12)

    # add colorbar
    cbar_ax = fig.add_axes([0.91, 0.67, 0.01, 0.2])
    cbar = plt.colorbar(im_dff, cax=cbar_ax)
    cbar.set_label('dF/F', fontsize=15)

    cbar_ax = fig.add_axes([0.91, 0.43, 0.01, 0.2])
    cbar = plt.colorbar(im_events, cax=cbar_ax)
    cbar.set_label('Events', fontsize=15)

    # suptitle
    fig.suptitle(f'Session # {session_row.session_ind} ({session_row.session_type} #{session_row.session_type_exposures})\n{session_row.stimulus} #{session_row.stimulus_exposures}\n{session_row.name}',
                fontsize=20, linespacing=1.7, y=0.96)
    fig.subplots_adjust(top=0.88, hspace=0.22)

    return fig, ax


def run_and_save_session_traces(session_info_df, save_dir,
                                dff_cmap='PRGn', event_vmax_percentile=80, event_vmax_adjust=1.0):
    # Save both dff and event traces
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # TODO: add responses to STAGE_*
    run_df = session_info_df[~session_info_df.session_type.str.contains('STAGE_')]
    
    for i in tqdm(range(len(run_df))):
        row = run_df.iloc[i]
        session_name = row['session_name']
        print(f'Processing {session_name} ({i+1}/{len(run_df)})')
        raw_path = row['raw_path']
        bod_list = cdu.get_bod_list(raw_path)
        dff_traces, event_traces, num_cells_per_plane, \
            running_speed, pupil, auto_rewards_mask, rewards_mask, \
                ophys_timestamps = get_session_traces(bod_list)

        fig, ax = plot_session_traces(dff_traces, event_traces, num_cells_per_plane,
                        running_speed, pupil, auto_rewards_mask, rewards_mask, 
                        ophys_timestamps, row, dff_cmap=dff_cmap,
                        event_vmax_percentile=event_vmax_percentile,
                        event_vmax_adjust=event_vmax_adjust)
        fig.savefig(save_dir / f'session_traces_{row.name}.png',
                    dpi=300, bbox_inches='tight',
                    transparent=False, facecolor='white')
        plt.close(fig)


## Plot mean responses to flashes
def plot_condition_v2_mean_responses(mean_response_df, data_type, image_names,
                                     session_row):
    condition_order = ['all-images', ['hit', 'miss'], 'omission',                   
                        *image_names,
                        *[[f'hit - {image_name}', f'miss - {image_name}'] for image_name in image_names]]

    timestamps = mean_response_df.timestamps.values[0]

    fig, ax = plt.subplots(figsize=(10,11))
    ax.axis('off')
    gs = gridspec.GridSpec(152, 144)

    hspace = 12
    gap = 8
    all_axes = []
    # first 3 - all-images, hit and miss, omissions
    for i in range(3):
        column_width = 37
        wspace = 12
        start = i * (column_width + wspace)
        end = start + column_width
        if i == 0:
            ax = fig.add_subplot(gs[:24, start : end])
        else:
            ax = fig.add_subplot(gs[:24, start : end],
                                sharey=all_axes[0], sharex=all_axes[0])
        all_axes.append(ax)
    # the rest 16 - individual images and changes (Smaller sizes)
    # 4 x 4
    for i in range(4):
        for j in range(4):
            column_width = 27
            wspace = 9
            start = j * (column_width + wspace)
            end = start + column_width
            ax = fig.add_subplot(gs[24 + (1+i) * hspace + i * 18 + gap : 42 + (1+i) * hspace + i * 18 + gap,
                                    start : end],
                                    sharey=all_axes[0], sharex=all_axes[0])
            all_axes.append(ax)

    assert len(all_axes) == len(condition_order)

    def _get_df_and_plot(ax, mean_response, dt, c, timestamps, color='k'):
        temp_df = mean_response.query(f'data_type=="{dt}" and condition=="{c}"')
        if temp_df.empty:
            return 0
        assert len(temp_df.num_incidents.unique()) == 1
        if (dt == 'dff') or (dt == 'events'):
            mean_trace = np.mean(np.vstack(temp_df.mean_trace.values), axis=0)
            sem_trace = np.std(np.vstack(temp_df.mean_trace.values), axis=0) / np.sqrt(len(temp_df))
        else:
            if len(temp_df) > 1:
                num_planes = len(mean_response.plane_name.unique())
                assert len(temp_df) == num_planes
            mean_trace = temp_df.mean_trace.values[0]
            sem_trace = temp_df.sem_trace.values[0]
        if ' - im' in c:
            label = None
        else:
            label = c
        ax.plot(timestamps, mean_trace, color=color, label=label)
        ax.fill_between(timestamps, mean_trace - sem_trace, mean_trace + sem_trace, color=color, alpha=0.5)
        return 1

    def _add_flashes(ax, timestamps, condition):
        if 'change' in condition:
            vba_tools.plot_flashes_on_trace(ax, timestamps, change_time=0, change=True,
                                            alpha=0.3)
        elif 'omission' in condition:
            vba_tools.plot_flashes_on_trace(ax, timestamps, change_time=0, omitted=True,
                                            alpha=0.3)
        else:
            vba_tools.plot_flashes_on_trace(ax, timestamps, change_time=0, alpha=0.3)

    for i, ax in enumerate(all_axes):
        ax = all_axes[i]
        condition = condition_order[i]
        
        if isinstance(condition, str):
            exists = _get_df_and_plot(ax, mean_response_df, data_type, condition, timestamps)
            if exists:
                _add_flashes(ax, timestamps, condition)
            ax.set_title(condition, fontsize=12)
        elif isinstance(condition, list):
            assert len(condition) == 2
            hit_cond_ind = np.where(['hit' in c for c in condition])[0]
            miss_cond_ind = np.where(['miss' in c for c in condition])[0]
            assert len(hit_cond_ind) == len(miss_cond_ind) == 1
            assert hit_cond_ind != miss_cond_ind
            hit_cond = condition[hit_cond_ind[0]]
            miss_cond = condition[miss_cond_ind[0]]
            _get_df_and_plot(ax, mean_response_df, data_type, hit_cond, timestamps, color='g')
            _get_df_and_plot(ax, mean_response_df, data_type, miss_cond, timestamps, color='m')
            _add_flashes(ax, timestamps, 'change')
            if ' - im' in hit_cond:
                image_name = hit_cond.split(' - ')[1]
                title = f'{image_name}-change'
                ax.set_title(title)
            else:
                # title = f'change\n '
                ax.legend(loc='lower left', frameon=False, fontsize=15, ncol=2, bbox_to_anchor=(-0.2, 0.9))
        else:
            raise ValueError(f'Invalid condition: {condition}')
        
        ax.set_xlim([-1, 1.5])
        
        if (i == 0) or (i == 15):
            ax.set_ylabel(data_type, fontsize=15)
        if i >= 15:
            ax.set_xlabel('Time (s)', fontsize=15)

    fig.suptitle(f'{session_row.name}\n{session_row.session_type} #{session_row.session_type_exposures}\n{data_type}',
                fontsize=15, linespacing=2, y=0.96)
    fig.subplots_adjust(top=0.82)

    return fig, ax


def run_and_save_mean_responses(session_info_df, save_dir, condition_version=2,
                                mean_response_dir=None):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if mean_response_dir is None:
        mouse_id = session_info_df.index[0].split('_')[0]
        mean_response_dir = f'/root/capsule/data/conditioned_mean_response_v{condition_version}_{mouse_id}_natural_image_sessions'
    if isinstance(mean_response_dir, str):
        mean_response_dir = Path(mean_response_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # TODO: add responses to gratings (TRAINING_* and STAGE_*)
    run_df = session_info_df[session_info_df.stimulus.str.contains('images_')]

    for i in tqdm(range(len(run_df))):
        row = run_df.iloc[i]
        session_name = row.session_name
        print(f'Processing {session_name} ({i+1}/{len(run_df)})')
        mean_response_path = mean_response_dir / f'response_v{condition_version}_{session_name}.feather'
        mean_response_df = pd.read_feather(mean_response_path)
        bod = cdu.get_any_bod(row.raw_path)
        image_names = np.sort(bod.stimulus_presentations.image_name[~bod.stimulus_presentations.image_name.isna()].unique())
        image_names = image_names[image_names != 'omitted']
        lamf_utils.condition_rename(mean_response_df, condition_version)

        for data_type in mean_response_df.data_type.unique():
            fig, _ = plot_condition_v2_mean_responses(mean_response_df,
                                                      data_type, image_names,
                                                      row)
            fig.savefig(save_dir / f'mean_responses_{row.name}_{data_type}.png',
                        dpi=600, bbox_inches='tight',
                        transparent=False, facecolor='white')
            plt.close(fig)
        

