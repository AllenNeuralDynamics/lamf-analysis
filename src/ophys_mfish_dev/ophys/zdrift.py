from pathlib import Path
import os
import h5py
import numpy as np
import json
from typing import Union
import skimage
import scipy
import pandas as pd
import cv2
from pystackreg import StackReg

import ophys_mfish_dev.utils as utils
import ophys_mfish_dev.ophys.zstack as zstack

###############################################################
# Zdrift 
#
# Use two-step registration approach:
# 1. Phase correlation to roughly match in x-y.
#   - In some cases x-y shift is too much that StackReg does not work right away.
# 2. StackReg to register the mean FOV to the z-stack
# Use CLAHE for contrast adjustment
# Median filtering and rolling averaging the z-stack
# Calculate correlation coefficient only in the valid pixels after transformation
# Penalize too much shift by using valid pixel threshold
# Consider using phase correlation for the first step registration
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
                use_valid_pix_pc=True, 
                use_valid_pix_sr=True):
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
    use_valid_pix_pc : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 1st step - phase correlation registration, by default True
    use_valid_pix_sr : bool, optional
        if to use valid pixels only for correlation coefficient calculation
        during the 2nd step - StackReg registration, by default True

    Returns
    -------
    np.ndarray (1d: int)
        matched plane indice from all the segments
    np.ndarray (2d: float)
        correlation coeffiecient for each segment across the reference z-stack
    np.ndarray (3d: float)
        Segment FOVs registered to the z-stack
    int
        Index of the matched plane for the mean FOV
    np.ndarray (1d: float)
        correlation coefficient for the mean FOV across the reference z-stack
    np.ndarray (2d: float)
        Mean FOV registered to the reference z-stack
    np.ndarray (3d: float)
        Reference z-stack cropped using motion output of the experiment
    np.ndarray (2d: float)
        Transformation matrix for StackReg RIGID_BODY
    np.ndarray (1d: int or float)
        An array of translation shift
    dict
        Options for calculating z-drift
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

    # Get episodic mean FOVs (emf) and crop
    emf_h5_fn = list(Path(movie_dir).glob('*_decrosstalk_episodic_mean_fov.h5'))[0]
    with h5py.File(emf_h5_fn, 'r') as h:
        episodic_mean_fovs = h['data'][:]
    episodic_mean_fovs_crop = episodic_mean_fovs[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    # Get first and last minute FOVs and crop
    processing_json_fn = list(Path(movie_dir).glob('processing.json'))[0]
    processing_json = json.load(open(processing_json_fn))
    frame_rate = processing_json['processing_pipeline']['data_processes'][0]['parameters']['movie_frame_rate_hz']
    movie_h5_fn = list(Path(movie_dir).glob('*_decrosstalk.h5'))[0]
    with h5py.File(movie_h5_fn, 'r') as h:
        first_minute = h['data'][:int(frame_rate * 60)]
        last_minute = h['data'][-int(frame_rate * 60):]
    first_minute_crop = first_minute[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]
    last_minute_crop = last_minute[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    # Run registration for each episodic mean FOVs
    matched_plane_indices = np.zeros(
        episodic_mean_fovs_crop.shape[0], dtype=int)
    corrcoef = []
    segment_reg_imgs = []
    sr_tmat_list = []
    translation_shift_list = []
    for i in range(episodic_mean_fovs_crop.shape[0]):
        mpi, cc, regimg, cc_pre, regimg_pre, sr_tmat, translation_shift = two_step_register_to_stack(
            episodic_mean_fovs_crop[i], ref_zstack_crop, use_clahe=use_clahe,
            use_valid_pix_pc=use_valid_pix_pc, use_valid_pix_sr=use_valid_pix_sr)
        matched_plane_indices[i] = mpi
        corrcoef.append(cc)
        segment_reg_imgs.append(regimg)
        sr_tmat_list.append(sr_tmat)
        translation_shift_list.append(translation_shift)
    corrcoef = np.asarray(corrcoef)

    emf_results = {'matched_plane_indices': matched_plane_indices,
                   'corrcoef': corrcoef,
                   'segment_fov_registered': segment_reg_imgs,
                   'corrcoef_pre': cc_pre,
                   'segment_fov_registered_pre': regimg_pre,
                   'ref_zstack_crop': ref_zstack_crop,
                   'sr_tmat': sr_tmat_list,
                   'translation_shift': translation_shift_list}
    
    # Run registration for the first and last minute FOVs
    first_minute_matched_plane, first_minute_corrcoef, first_minute_regimg, first_minute_tmat, _ = two_step_register_to_stack(
        first_minute_crop.mean(axis=0), ref_zstack_crop, use_clahe=use_clahe,
        use_valid_pix_pc=use_valid_pix_pc, use_valid_pix_sr=use_valid_pix_sr)
    last_minute_matched_plane, last_minute_corrcoef, last_minute_regimg, last_minute_tmat, _ = two_step_register_to_stack(
        last_minute_crop.mean(axis=0), ref_zstack_crop, use_clahe=use_clahe,
        use_valid_pix_pc=use_valid_pix_pc, use_valid_pix_sr=use_valid_pix_sr)
    
    first_last_results = {'first_minute_matched_plane': first_minute_matched_plane,
                          'first_minute_corrcoef': first_minute_corrcoef,
                          'first_minute_regimg': first_minute_regimg,
                          'first_minute_tmat': first_minute_tmat,
                          'last_minute_matched_plane': last_minute_matched_plane,
                          'last_minute_corrcoef': last_minute_corrcoef,
                          'last_minute_regimg': last_minute_regimg,
                          'last_minute_tmat': last_minute_tmat}
    
    ops = {'use_clahe': use_clahe,
           'use_valid_pix_pc': use_valid_pix_pc,
           'use_valid_pix_sr': use_valid_pix_sr}

    # if save_data:
    #     # Save the result
    #     if not os.path.isdir(exp_dir):
    #         os.makedirs(exp_dir)
    #     with h5py.File(exp_dir / exp_fn, 'w') as h:
    #         h.create_dataset('matched_plane_indices',
    #                             data=matched_plane_indices)
    #         h.create_dataset('corrcoef', data=corrcoef)
    #         h.create_dataset('segment_fov_registered',
    #                             data=segment_reg_imgs)
    #         h.create_dataset('corrcoef_pre', data=cc_pre)
    #         h.create_dataset('segment_fov_registered_pre',
    #                             data=regimg_pre)
    #         h.create_dataset('ref_oeid', data=ref_oeid)
    #         h.create_dataset('ref_zstack_crop', data=ref_zstack_crop)
    #         h.create_dataset('rigid_tmat', data=rigid_tmat_list)
    #         h.create_dataset('translation_shift',
    #                             data=translation_shift_list)
    #         h.create_dataset('ops/use_clahe', shape=(1,), data=use_clahe)
    #         h.create_dataset('ops/use_valid_pix_pc',
    #                             shape=(1,), data=use_valid_pix_pc)
    #         h.create_dataset('ops/use_valid_pix_sr',
    #                             shape=(1,), data=use_valid_pix_sr)

    return emf_results, first_last_results, ops


####################################################################################################
# FOV + Zstack registration
####################################################################################################

def two_step_register_to_stack(fov, stack, use_clahe=True, use_valid_pix_pc=False, use_valid_pix_sr=True,
                               sr_method='affine'):
    """Register FOV to z-stack using 2-step registration
    First register using phase correlation and then using StackReg

    Parameters
    ----------
    fov : np.ndarray (2d)
        FOV image
    zstack : np.ndarray (3d)
        z-stack images
    use_clahe : bool, optional
        If to adjust contrast using CLAHE for registration, by default True
    use_valid_pix_pc : bool, optional
        If to use valid pixels (non-blank pixels after transfromation)
        to calculate correlation coefficient during the 1st step registration, by default True
    use_valid_pix_sr : bool, optional
        If to use valid pixels (non-blank pixels after transfromation)
        to calculate correlation coefficient during the 2nd step registration, by default True
    sr_method : str, optional
        StackReg method, by default 'affine'

    Returns
    -------
    np.ndarray (1d: int)
        matched plane index
    np.ndarray (2d: float)
        correlation coefficient for each plane in the z-stack
    np.ndarray (2d: float)
        registered FOV
    np.ndarray (2d: float)
        transformation matrix for StackReg RIGID_BODY
    np.ndarray (1d: int or float)
        translation shift
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

    fov_reg_pc, corrcoef_pc_arr, shift_list = fov_stack_register_phase_correlation(
        fov_for_reg, stack_for_reg, use_clahe=False, use_valid_pix=use_valid_pix_pc)
    best_match_pc_ind = np.argmax(corrcoef_pc_arr)
    fov_for_stackreg = fov_reg_pc[best_match_pc_ind]
    best_shift = shift_list[best_match_pc_ind]
    corrcoef_sr_arr, fov_reg, best_tmat, tmat_list = fov_stack_register_stackreg(
        fov_for_stackreg, stack_for_reg, use_clahe=False, sr_method=sr_method, use_valid_pix=use_valid_pix_sr)
    matched_plane_index = np.argmax(corrcoef_sr_arr)
    return matched_plane_index, corrcoef_sr_arr, corrcoef_pc_arr, fov_reg, fov_for_stackreg, best_tmat, best_shift


def fov_stack_register_stackreg(fov, stack, use_clahe=True, sr_method='affine', tmat=None, use_valid_pix=True):

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
    
    if sr_method == 'affine':
        sr = StackReg(StackReg.AFFINE)
    elif sr_method == 'rigid_body':
        sr = StackReg(StackReg.RIGID_BODY)
    else:
        raise ValueError('"sr_method" should be either "affine" or "rigid_body"')
    
    assert fov.min() >= 0
    if use_valid_pix:
        valid_pix_threshold = fov.min()/10
    else:
        valid_pix_threshold = -1
    num_pix_threshold = fov.shape[0] * fov.shape[1] / 2
    
    corrcoef_arr = np.zeros(stack.shape[0])
    
    if tmat is None:
        temp_cc = []
        tmat_list = []
        for zi in range(stack_for_reg.shape[0]):
            zstack_plane_clahe = stack_for_reg[zi]
            zstack_plane = stack[zi]
            tmat = sr.register(zstack_plane_clahe, fov_for_reg)
            fov_reg = sr.transform(fov, tmat=tmat)            
            valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
            if len(valid_y) > num_pix_threshold:
                temp_cc.append(np.corrcoef(zstack_plane[valid_y, valid_x].flatten(),
                                           fov_reg[valid_y, valid_x].flatten())[0,1])
                tmat_list.append(tmat)
            else:
                temp_cc.append(0)
                tmat_list.append(np.eye(3))
        temp_ind = np.argmax(temp_cc)
        best_tmat = tmat_list[temp_ind]
    else:
        best_tmat = tmat
    fov_reg = sr.transform(fov, tmat=best_tmat)
    for zi, zstack_plane in enumerate(stack):
        corrcoef_arr[zi] = np.corrcoef(zstack_plane[valid_y, valid_x].flatten(),
                                   fov_reg[valid_y, valid_x].flatten())[0,1]
    return corrcoef_arr, fov_reg, best_tmat, tmat_list    


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

    # stack_pre = med_filt_z_stack(stack)
    # stack_pre = rolling_average_stack(stack_pre)

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

    fov_reg_stack = np.zeros_like(stack)
    corrcoef_arr = np.zeros(stack.shape[0])
    shift_list = []
    for pi in range(stack.shape[0]):
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


# TODO: finish refactor
# def fov_stack_register_rigid(fov, stack, use_valid_pix=True):
#     """Find the best-matched plane index from the stack using rigid-body registration (StackReg)
#     First register the FOV to each plane in the stack
#     First pass: Pick the best-matched plane using pixel-wise correlation
#     Second pass: Using the transformed FOV to the first pass best-matched plane,
#         sweep through all planes and refine the best-matched plane using correlation.
#         This fixes minor mis-match due to registration error.

#     Parameters
#     ----------
#     fov : np.ndarray (2D)
#         input FOV
#     stack : np.ndarray (3D)
#         stack images
#     use_valid_pix : bool, optional
#         If to use valid pixels (non-blank pixels after transfromation)
#         to calculate correlation coefficient, by default True

#     Returns
#     -------
#     np.ndarray (1D)
#         matched plane index

#     """
#     assert len(fov.shape) == 2
#     assert len(stack.shape) == 3
#     assert fov.shape == stack.shape[1:]

#     # TODO: figure out what is the best threshold. This value should be larger than one because of the results after registration
#     valid_pix_threshold = 10  # for uint16 data

#     stack_pre = med_filt_z_stack(stack)
#     stack_pre = rolling_average_stack(stack_pre)

#     sr = StackReg(StackReg.RIGID_BODY)

#     # apply CLAHE and normalize to make it uint16 (for StackReg)
#     fov_clahe = image_normalization(skimage.exposure.equalize_adapthist(
#         fov.astype(np.uint16)),dtype='uint16')  # normalization to make it uint16
#     stack_clahe = np.zeros_like(stack_pre)
#     for pi in range(stack_pre.shape[0]):
#         stack_clahe[pi, :, :] = image_normalization(
#             skimage.exposure.equalize_adapthist(stack_pre[pi, :, :].astype(np.uint16)),dtype='uint16')

#     # Initialize
#     corrcoef = np.zeros(stack_clahe.shape[0])
#     fov_reg_stack = np.zeros_like(stack)
#     tmat_list = []
#     for pi in range(len(corrcoef)):
#         tmat = sr.register(stack_clahe[pi, :, :], fov_clahe)
#         # Apply the transformation matrix to the FOV registered using phase correlation
#         fov_reg = sr.transform(fov, tmat=tmat)
#         if use_valid_pix:
#             valid_y, valid_x = np.where(fov_reg > valid_pix_threshold)
#             corrcoef[pi] = np.corrcoef(
#                 stack_pre[pi, valid_y, valid_x].flatten(), fov_reg[valid_y, valid_x].flatten())[0, 1]
#         else:
#             corrcoef[pi] = np.corrcoef(
#                 stack_pre[pi, :, :].flatten(), fov_reg.flatten())[0, 1]
#         fov_reg_stack[pi, :, :] = fov_reg
#         tmat_list.append(tmat)
#     matched_plane_index = np.argmax(corrcoef)
#     tmat = tmat_list[matched_plane_index]

#     return matched_plane_index, corrcoef, tmat


# TODO: finish refactor
# def get_segment_mean_images(ophys_experiment_id, save_dir=None, save_images=True, segment_minute=10):
#     """Get segmented mean images of an experiment
#     And save the result, in case the directory was specified.

#     Parameters
#     ----------
#     ophys_experiment_id : int
#         ophys experiment ID
#     save_dir : Path, optional
#         ophys experiment directory to load/save.
#         If None, then do not attemp loading previously saved data (it takes 2-3 min)
#     segment_minute : int, optional
#         length of a segment, in min, by default 10

#     Returns
#     -------
#     np.ndarray (3d)
#         mean FOVs from each segment of the video
#     """

#     if save_dir is None:
#         save_dir = global_base_dir
#     got_flag = 0
#     segment_fov_fp = save_dir / f'{ophys_experiment_id}_segment_fov.h5'
#     if os.path.isfile(segment_fov_fp):
#         with h5py.File(segment_fov_fp, 'r') as h:
#             mean_images = h['data'][:]
#             got_flag = 1

#     if got_flag == 0:
#         frame_rate, timestamps = get_correct_frame_rate(ophys_experiment_id)
#         movie_fp = from_lims.get_motion_corrected_movie_filepath(
#             ophys_experiment_id)
#         h = h5py.File(movie_fp, 'r')
#         movie_len = h['data'].shape[0]
#         segment_len = int(np.round(segment_minute * frame_rate * 60))

#         # get frame start and end (for range) indices
#         frame_start_end = get_frame_start_end(movie_len, segment_len)

#         mean_images = np.zeros((len(frame_start_end), *h['data'].shape[1:]))
#         for i in range(len(frame_start_end)):
#             mean_images[i, :, :] = np.mean(
#                 h['data'][frame_start_end[i][0]:frame_start_end[i][1], :, :], axis=0)
#         if save_images:
#             if not os.path.isdir(save_dir):
#                 os.makedirs(save_dir)
#             with h5py.File(segment_fov_fp, 'w') as h:
#                 h.create_dataset('data', data=mean_images)
#     return mean_images

