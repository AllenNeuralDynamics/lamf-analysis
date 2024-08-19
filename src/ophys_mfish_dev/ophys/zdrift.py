from pathlib import Path
import h5py
import numpy as np
from typing import Union
import skimage
import scipy
import cv2

import ophys_mfish_dev.utils as utils
import ophys_mfish_dev.ophys.zstack as zstack

###############################################################
# Zdrift 
#
# Use phase correlation:
# Median filtering and rolling averaging the z-stack
# Crop images based on the motion correction output
# (Optional) Use CLAHE for contrast adjustment
# (Optional) Calculate correlation coefficient only in the valid pixels after transformation
#       - Penalize too much shift by using valid pixel threshold
# Session to session matching by using zstack to zstack registration (implemented in a different file) 
###############################################################

def zdrift_for_session_planes(session_path: Union[Path, str],
                              **zdrift_kwargs) -> dict:
    """Get z-drift for all the planes in a session
    Parameters
    ----------
    session_path : Path
        Path to the session directory
    zdrift_kwargs : dict
        Arguments for calc_zdrift

    Returns
    -------
    dict
        Dictionary of z-drift for each plane
    """
    path_to_all_planes = utils.plane_paths_from_session(session_path, data_level="processed")

    zdrift_dict = {}
    for path_to_plane in path_to_all_planes:
        plane_id = int(path_to_plane.name.split('_')[-1])
        zdrift_dict[plane_id] = calc_zdrift(path_to_plane, **zdrift_kwargs)
    return zdrift_dict


def calc_zdrift(movie_dir: Path, 
                reference_stack_path: Path,
                use_clahe=True, 
                use_valid_pix=True, 
                ):
    """Calc zdrift for an ophys movie relative to reference stack

    Register the mean FOV image to the z-stack using 2-step registration.
    Then, apply the same transformation to segmented FOVs and
    calculated matched planes in each segment.

    Add first minute and last minute movie as well.
    Movie should be out of decrosstalk directory, if multiscope is used.
    Otherwise, it should be from motion correction directory.
    # TODO: Define movie directory based on the rig or operation mode (06/2024)

    Parameters
    ----------
    movie_dir : Path
        Path to the movie directory (should be "decrosstalk" dir)
    reference_stack_path : Path
        Path to the reference stack
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

    save_dir = Path(save_dir)

    # to remove rolling effect from motion correction
    range_y, range_x = utils.get_motion_correction_crop_xy_range(
        movie_dir)  

    # Get reference z-stack and crop
    # TODO: it should be processed first in the pipeline, and this part of code 
    # should just retrieve the processed (registered) z-stack
    ref_zstack = zstack.register_local_z_stack(reference_stack_path)
    ref_zstack_crop = ref_zstack[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    # Get preprocessed z-stack
    stack_pre = med_filt_z_stack(ref_zstack_crop)
    stack_pre = rolling_average_stack(stack_pre)

    # Get episodic mean FOVs (emf) and crop
    emf_h5_fn = list(Path(movie_dir).glob('*_decrosstalk_episodic_mean_fov.h5'))[0]
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
        fov_reg, cc, shift = fov_stack_register_phase_correlation(
            episodic_mean_fovs_crop[i], stack_pre, use_clahe=use_clahe,
            use_valid_pix=use_valid_pix)
        matched_plane_indices[i] = np.argmax(cc)
        corrcoef.append(cc)
        segment_reg_imgs.append(fov_reg)
        shift_list.append(shift)
    corrcoef = np.asarray(corrcoef)

    results = {'matched_plane_indices': matched_plane_indices,
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
        list of translation shifts
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