from pathlib import Path
import h5py
import numpy as np
from typing import Union
import skimage
import scipy
import cv2
import matplotlib.pyplot as plt
import ray
import sys
import os
os.environ["RAY_verbose_spill_logs"] = "0"

import lamf_analysis.utils as utils
import lamf_analysis.ophys.zstack as zstack

###############################################################
# Zdrift 
#
# Use phase correlation:
# Median filtering and rolling averaging the z-stack
# Crop images based on the motion correction output
# Take mean FOV image, calculate transformation, apply to episodic mean FOVs, and calculate correlation coefficient
# (Optional) Use CLAHE for contrast adjustment
# (Optional) Calculate correlation coefficient only in the valid pixels after transformation
#       - Penalize too much shift by using valid pixel threshold
# Session to session matching by using zstack to zstack registration (implemented in a different file) 
###############################################################

def zdrift_for_session_planes(raw_path: Union[Path, str],
                              parallel: bool = True,
                              **zdrift_kwargs) -> dict:
    """Get z-drift for all the planes in a session
    Parameters
    ----------
    raw_path : Path
        Path to the raw session directory
    zdrift_kwargs : dict
        Arguments for calc_zdrift (use_clahe and use_valid_pix)

    Returns
    -------
    dict
        Dictionary of z-drift for each plane
    """
    raw_path_to_all_planes = utils.plane_paths_from_session(raw_path,
                                                        data_level="raw")

    if parallel:
        spill_dir = "/root/capsule/scratch/ray"
        sys.path.append('/root/capsule/code')  
        ray.init(ignore_reinit_error=True,
                _temp_dir=spill_dir,
                object_store_memory=(2**10)**3 * 4,
                _system_config={"object_spilling_config": f'{{"type":"filesystem","params":{{"directory_path":"{spill_dir}"}}}}'},
                runtime_env={"working_dir": "/root/capsule/code",
                             "excludes": list(Path('/root/capsule/code').rglob('*.ipynb'))})
        futures = []
        for path_to_plane in raw_path_to_all_planes:
            futures.append(ray.remote(calc_zdrift).remote(path_to_plane, **zdrift_kwargs))
        result_dict = ray.get(futures)
        ray.shutdown()
        plane_ids = np.sort([result['plane_id'] for result in result_dict])
        zdrift_dict = {}
        for plane_id in plane_ids:
            result = [result for result in result_dict if result['plane_id'] == plane_id][0]
            zdrift_dict[plane_id] = result
    else:
        zdrift_dict = {}
        for path_to_plane in raw_path_to_all_planes:
            plane_id = int(path_to_plane.name.split('_')[-1])
            zdrift_dict[plane_id] = calc_zdrift(path_to_plane, **zdrift_kwargs)
    return zdrift_dict


def calc_zdrift(raw_plane_path: Path,
                use_clahe=True, 
                use_valid_pix=True, 
                ):
    """Calc zdrift for an ophys movie relative to reference stack

    Register the mean FOV image to the z-stack.
    Then, apply the same transformation to segmented FOVs and
    calculated matched planes in each segment.

    Movie should be out of decrosstalk directory, if multiscope is used.
    Otherwise, it should be from motion correction directory.
    # TODO: Define movie directory based on the rig or operation mode (06/2024)

    Parameters
    ----------
    raw_plane_path : Path
        Path to the raw plane directory
    save_dir : Path
        Path to save the result
    segment_minute : int, optional
        Number of minutes to segment the plane video, by default 10
    correct_image_size : tuple, optional
        Tuple for correct image size, (y,x), by default (512,512)
    use_clahe : bool, optional
        if to use CLAHE for registration, by default True
    use_valid_pix : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 1st step - phase correlation registration, by default True

    Returns
    -------
    dict
        Results and options for calculating z-drift
    """
    # valid_threshold = 0.01  # TODO: find the best valid threshold
    # find processed plane path from raw plane path
    if isinstance(raw_plane_path, str):
        raw_plane_path = Path(raw_plane_path)
    plane_id = raw_plane_path.stem
    session_name = raw_plane_path.parent.parent.stem
    processed_path = list(raw_plane_path.parent.parent.parent.glob(f'{session_name}_processed_*'))
    assert len(processed_path) == 1
    processed_path = processed_path[0]
    processed_plane_path = processed_path / plane_id

    # to remove rolling effect from motion correction
    range_y, range_x = utils.get_motion_correction_crop_xy_range(
        processed_plane_path)

    # Get reference z-stack and crop
    # TODO: it should be processed first in the pipeline, and this part of code 
    # should just retrieve the processed (decrosstalked & registered) z-stack

    # TODO: the following code does not work with data uploaded from rig
    # z-stack splitting and saving to h5 should be done first
    try:
        local_zstack_path = list(raw_plane_path.glob('*_z_stack_local.h5'))[0]
    except:
        raise FileNotFoundError('Local z-stack not found')
    ref_zstack = zstack.register_local_z_stack(local_zstack_path)
    ref_zstack_crop = ref_zstack[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    si_metadata, roi_groups = zstack.local_zstack_metadata(local_zstack_path)
    number_of_z_planes= int(si_metadata['SI.hStackManager.actualNumSlices'])
    # number_of_repeats = int(si_metadata['SI.hStackManager.actualNumVolumes'])
    z_step = float(si_metadata['SI.hStackManager.actualStackZStepSize'])

    # Get preprocessed z-stack
    stack_pre = med_filt_z_stack(ref_zstack_crop)
    stack_pre = rolling_average_stack(stack_pre)

    # Get episodic mean FOVs (emf) and crop
    # TODO: make the mean FOV movie with finer time resolution
    decrosstalk_dir = processed_plane_path / 'decrosstalk'
    emf_h5_fn = list(Path(decrosstalk_dir).glob('*_decrosstalk_episodic_mean_fov.h5'))[0]
    with h5py.File(emf_h5_fn, 'r') as h:
        episodic_mean_fovs = h['data'][:]
    episodic_mean_fovs_crop = episodic_mean_fovs[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    # Run registration for each episodic mean FOVs
    matched_plane_indices = np.zeros(
        episodic_mean_fovs_crop.shape[0], dtype=int)
    corrcoef = []
    segment_reg_imgs = []
    shift_list = []
    for i in range(episodic_mean_fovs_crop.shape[0]):
        fov_reg_stack, cc, shift = fov_stack_register_phase_correlation(
            episodic_mean_fovs_crop[i], stack_pre, use_clahe=use_clahe,
            use_valid_pix=use_valid_pix)
        matched_plane_indices[i] = np.argmax(cc)
        corrcoef.append(cc)
        segment_reg_imgs.append(fov_reg_stack[np.argmax(cc)])
        shift_list.append(shift)
    corrcoef = np.asarray(corrcoef)

    center_z = number_of_z_planes // 2
    zdrift_um = z_step * (matched_plane_indices - center_z)

    results = {'plane_id': plane_id,
                'zdrift_um': zdrift_um,
                'matched_plane_indices': matched_plane_indices,
                'corrcoef': corrcoef,
                'segment_fov_registered': segment_reg_imgs,
                'ref_zstack_crop': ref_zstack_crop,                   
                'shift': shift_list,
                'use_clahe': use_clahe,
                'use_valid_pix': use_valid_pix}

    return results


def fov_stack_register_phase_correlation(fov, stack, use_clahe=True, use_valid_pix=True):
    """ Reigster FOV to each plane in the stack

    Parameters
    ----------
    fov : np.ndarray (2d)
        FOV image
    stack : np.ndarray (3d)
        stack images
    use_clahe: bool, optional
        If to adjust contrast using CLAHE for registration, by default True
    use_valid_pix : bool, optional
        If to use valid pixels (non-blank pixels after transfromation)
        to calculate correlation coefficient, by default True

    Returns
    -------
    np.ndarray (3d)
        stack of FOV registered to each plane in the input stack
    np.array (1d)
        correlation coefficient between the registered fov and the stack in each plane
    list
        list of translation shifts (y,x)
    """
    assert len(fov.shape) == 2
    assert len(stack.shape) == 3
    assert fov.shape == stack.shape[1:]

    if use_clahe:
        fov_for_reg = image_normalization(skimage.exposure.equalize_adapthist(
            fov.astype(np.uint16)))  # normalization to make it uint16
        stack_for_reg = np.zeros_like(stack)
        for pi in range(stack.shape[0]):
            stack_for_reg[pi, :, :] = image_normalization(
                skimage.exposure.equalize_adapthist(stack[pi, :, :].astype(np.uint16)))
    else:
        fov_for_reg = fov.copy()
        stack_for_reg = stack.copy()

    fov_reg_stack = np.zeros_like(stack_for_reg)
    corrcoef_arr = np.zeros(stack_for_reg.shape[0])
    shift_list = []
    for pi in range(stack_for_reg.shape[0]):
        shift, _, _ = skimage.registration.phase_cross_correlation(
            stack_for_reg[pi, :, :], fov_for_reg, normalization=None)
        fov_reg = scipy.ndimage.shift(fov, shift)
        fov_reg_stack[pi, :, :] = fov_reg
        if use_valid_pix:
            valid_y, valid_x = np.where(fov_reg > 0)
            corrcoef_arr[pi] = np.corrcoef(stack[pi, valid_y, valid_x].flatten(
            ), fov_reg[valid_y, valid_x].flatten())[0, 1]
        else:
            corrcoef_arr[pi] = np.corrcoef(
                stack[pi, :, :].flatten(), fov_reg.flatten())[0, 1]
        shift_list.append(shift)    
    return fov_reg_stack, corrcoef_arr, shift_list


def med_filt_z_stack(zstack, kernel_size=5):
    """Get z-stack with each plane median-filtered

    Parameters
    ----------
    zstack : np.ndarray
        z-stack to apply median filtering
    kernel_size : int, optional
        kernel size for median filtering, by default 5
        It seems only certain odd numbers work, e.g., 3, 5, 11, ...

    Returns
    -------
    np.ndarray
        median-filtered z-stack
    """
    filtered_z_stack = []
    for image in zstack:
        filtered_z_stack.append(cv2.medianBlur(
            image.astype(np.uint16), kernel_size))
    return np.array(filtered_z_stack)


def rolling_average_stack(stack, rolling_window_flank=2):
    """ Get a stack with each plane rolling-averaged
    
    Parameters
    ----------
    stack : np.ndarray
        stack to apply rolling average
    rolling_window_flank : int, optional
        flank size for rolling average, by default 2
        
    Returns
    -------
    np.ndarray
        rolling-averaged stack
    """
    new_stack = np.zeros(stack.shape)
    for i in range(stack.shape[0]):
        new_stack[i] = np.mean(stack[max(0, i-rolling_window_flank) : min(stack.shape[0], i+rolling_window_flank), :, :],
                                axis=0)
    return new_stack


def image_normalization(image, im_thresh=0, dtype=np.uint16):
    """Normalize 2D image and convert to specified dtype
    Prevent saturation.

    Args:
        image (np.ndarray): input image (2D)
                            Just works with 3D data as well.
        im_thresh (float, optional): threshold when calculating pixel intensity percentile.
                            0 by default
        dtype (np.dtype, optional): output data type. np.uint16 by default
    Return:
        norm_image (np.ndarray)
    """
    clip_image = np.clip(image, np.percentile(
        image[image > im_thresh], 0.2), np.percentile(image[image > im_thresh], 99.8))
    norm_image = (clip_image - np.amin(clip_image)) / \
        (np.amax(clip_image) - np.amin(clip_image)) * 0.9
    image_dtype = ((norm_image + 0.05) *
                    np.iinfo(np.uint16).max * 0.9).astype(dtype)
    return image_dtype



###############################################################
## QC plots for z-drift
def plot_session_zdrift(result, ax=None, cc_threshold=0.65):
    """Plot z-drift for all the segments in a session
    Drift with peak correlation coefficient overlaid

    Parameters
    ----------
    result : dict
        Dictionary of z-drift results for each plane
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    zdrift_um = result['zdrift_um']
    max_cc = np.array([max(cc) for cc in result['corrcoef']])

    # # test
    # max_cc = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.7, 0.8, 0.9, 0.95])
    
    ax.plot(zdrift_um, color='black', zorder=1)
    
    # split color by correlation coefficient
    h1 = ax.scatter(np.arange(len(max_cc)), zdrift_um, c=max_cc, s=50,
                cmap='binary', vmin=cc_threshold, vmax=1,
                edgecolors='black', linewidth=0.5, zorder=2)
    under_threshold_ind = np.where(max_cc < cc_threshold)[0]
    if len(under_threshold_ind) > 0:
        has_low_cc = 1
        h2 = ax.scatter(under_threshold_ind, zdrift_um[under_threshold_ind], s=50,
                        c=max_cc[under_threshold_ind], cmap='Reds_r', vmin=0, vmax=cc_threshold,
                        edgecolors='red', linewidth=0.5, zorder=3)
    else:
        ylim = ax.get_ybound()
        xlim = ax.get_xbound()
        h2 = ax.scatter(xlim[0] - 1, ylim[0] - 1, c=0,
                        cmap='Reds_r', vmin=0, vmax=cc_threshold)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    cax1 = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0 + (ax.get_position().height) * cc_threshold,
                        0.02,
                        ax.get_position().height * (1 - cc_threshold)])
    bar1 = plt.colorbar(h1, cax=cax1)
    bar1.set_label('Correlation coefficient')
    bar1.ax.yaxis.set_label_coords(6, -0.5)

    cax2 = fig.add_axes([ax.get_position().x1 + 0.01,
                    ax.get_position().y0,
                    0.02,
                    ax.get_position().height * cc_threshold])
    plt.colorbar(h2, cax=cax2)
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Z-drift (um)')
    plt.show()
    return ax


def plot_shifts(result, ax=None):
    """Plot shifts at the matched depth for all the segments in a session
    Both in x-y

    Parameters
    ----------
    result : dict
        Dictionary of z-drift results for each plane
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    max_cc_inds = np.array([np.argmax(cc) for cc in result['corrcoef']])        
    shifts = [result['shift'][i][max_cc_inds[i]] for i in range(len(max_cc_inds))]
    y_shift = [shift[0] for shift in shifts]
    x_shift = [shift[1] for shift in shifts]
    ax.plot(y_shift, color='c', label='y-shift')
    ax.plot(x_shift, color='m', label='x-shift')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Shift (pix)')
    ax.set_ylim(-512, 512)
    ax.legend()
    plt.show()
    return ax


def plot_correlation_coefficients(result, ax=None):
    """Plot correlation coefficients for all the segments in a session

    Parameters
    ----------
    result : dict
        Dictionary of z-drift results for each plane
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for i, cc in enumerate(result['corrcoef']):
        ax.plot(cc, label=f'Seg #{i}')
    ax.set_xlabel('Zstack plane index')
    ax.set_ylabel('Correlation coefficient')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    return ax