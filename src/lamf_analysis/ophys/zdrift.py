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


import ophys_mfish_dev.utils as utils
import ophys_mfish_dev.ophys.zstack as zstack

###############################################################
# Zdrift 
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
    planes = utils.plane_paths_from_session(session_path, data_level="processed")

    zdrift_dict = {}
    for plane in planes:
        plane_id = int(plane.name.split('_')[-1])
        zdrift_dict[plane_id] = calc_zdrift(plane, **zdrift_kwargs)
    return zdrift_dict


# TODO: not finished refactoring (04/30/24)
# def calc_zdrift(movie_path: Path, 
#                 reference_stack_path: Path, 
#                 segment_minute: int = 10, 
#                 correct_image_size=(512, 512),
#                 use_clahe=True, 
#                 use_valid_pix_pc=True, 
#                 use_valid_pix_sr=True):
#     """Calc zdrift for an ophys movie relative to reference stack

#     Register the mean FOV image to the z-stack using 2-step registration.
#     Then, apply the same transformation to segmented FOVs and
#     calculated matched planes in each segment.

#     Parameters
#     ----------
#     movie_path : Path
#         Path to the movie directory
#     reference_stack_path : Path
#         Path to the reference stack
#     segment_minute : int, optional
#         Number of minutes to segment the plane video, by default 10
#     correct_image_size : tuple, optional
#         Tuple for correct image size, (y,x), by default (512,512)
#     use_clahe : bool, optional
#         if to use CLAHE for registration, by default True
#     use_valid_pix_pc : bool, optional
#         if to use valid pixels only for correlation coefficient calculation
#         during the 1st step - phase correlation registration, by default True
#     use_valid_pix_sr : bool, optional
#         if to use valid pixels only for correlation coefficient calculation
#         during the 2nd step - StackReg registration, by default True

#     Returns
#     -------
#     np.ndarray (1d: int)
#         matched plane indice from all the segments
#     np.ndarray (2d: float)
#         correlation coeffiecient for each segment across the reference z-stack
#     np.ndarray (3d: float)
#         Segment FOVs registered to the z-stack
#     int
#         Index of the matched plane for the mean FOV
#     np.ndarray (1d: float)
#         correlation coefficient for the mean FOV across the reference z-stack
#     np.ndarray (2d: float)
#         Mean FOV registered to the reference z-stack
#     np.ndarray (3d: float)
#         Reference z-stack cropped using motion output of the experiment
#     np.ndarray (2d: float)
#         Transformation matrix for StackReg RIGID_BODY
#     np.ndarray (1d: int or float)
#         An array of translation shift
#     dict
#         Options for calculating z-drift
#     """
#     # valid_threshold = 0.01  # TODO: find the best valid threshold

#     save_dir = Path(save_dir)

#     # to remove rolling effect from motion correction
#     range_y, range_x = get_motion_correction_crop_xy_range(
#         oeid)  

#     # Get reference z-stack and crop
#     ref_zstack = register_local_z_stack(reference_stack_path)
#     ref_zstack_crop = ref_zstack[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

#     # Get segmented FOVs and crop
#     segment_mean_images = get_segment_mean_images(
#         oeid, save_dir=exp_dir, segment_minute=segment_minute)
#     segment_mean_images_crop = segment_mean_images[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

#     # Rung registration for each segmented FOVs
#     matched_plane_indices = np.zeros(
#         segment_mean_images_crop.shape[0], dtype=int)
#     corrcoef = []
#     segment_reg_imgs = []
#     rigid_tmat_list = []
#     translation_shift_list = []
#     for i in range(segment_mean_images_crop.shape[0]):
#         mpi, cc, regimg, cc_pre, regimg_pre, rigid_tmat, translation_shift = estimate_plane_from_ref_zstack(
#             segment_mean_images_crop[i], ref_zstack_crop, use_clahe=use_clahe,
#             use_valid_pix_pc=use_valid_pix_pc, use_valid_pix_sr=use_valid_pix_sr)
#         matched_plane_indices[i] = mpi
#         corrcoef.append(cc)
#         segment_reg_imgs.append(regimg)
#         rigid_tmat_list.append(rigid_tmat)
#         translation_shift_list.append(translation_shift)
#     corrcoef = np.asarray(corrcoef)

#     ops = {'use_clahe': use_clahe,
#             'use_valid_pix_pc': use_valid_pix_pc,
#             'use_valid_pix_sr': use_valid_pix_sr}

#     if save_data:
#         # Save the result
#         if not os.path.isdir(exp_dir):
#             os.makedirs(exp_dir)
#         with h5py.File(exp_dir / exp_fn, 'w') as h:
#             h.create_dataset('matched_plane_indices',
#                                 data=matched_plane_indices)
#             h.create_dataset('corrcoef', data=corrcoef)
#             h.create_dataset('segment_fov_registered',
#                                 data=segment_reg_imgs)
#             h.create_dataset('corrcoef_pre', data=cc_pre)
#             h.create_dataset('segment_fov_registered_pre',
#                                 data=regimg_pre)
#             h.create_dataset('ref_oeid', data=ref_oeid)
#             h.create_dataset('ref_zstack_crop', data=ref_zstack_crop)
#             h.create_dataset('rigid_tmat', data=rigid_tmat_list)
#             h.create_dataset('translation_shift',
#                                 data=translation_shift_list)
#             h.create_dataset('ops/use_clahe', shape=(1,), data=use_clahe)
#             h.create_dataset('ops/use_valid_pix_pc',
#                                 shape=(1,), data=use_valid_pix_pc)
#             h.create_dataset('ops/use_valid_pix_sr',
#                                 shape=(1,), data=use_valid_pix_sr)

#     return matched_plane_indices, corrcoef, segment_reg_imgs, \
#         ref_oeid, ref_zstack_crop, rigid_tmat_list, translation_shift_list, ops


####################################################################################################
# FOV + Zstack registration
####################################################################################################

# TODO: finish Refactor (04/2024)
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

    stack_pre = med_filt_z_stack(stack)
    stack_pre = rolling_average_stack(stack_pre)

    if use_clahe:
        fov_for_reg = image_normalization(skimage.exposure.equalize_adapthist(
            fov.astype(np.uint16)), dtype='unit16')  # normalization to make it uint16
        stack_for_reg = np.zeros_like(stack_pre)
        for pi in range(stack.shape[0]):
            stack_for_reg[pi, :, :] = image_normalization(
                skimage.exposure.equalize_adapthist(stack_pre[pi, :, :].astype(np.uint16)),dtype='uint16')
    else:
        fov_for_reg = fov.copy()
        stack_for_reg = stack_pre.copy()

    fov_reg_stack = np.zeros_like(stack)
    corrcoef = np.zeros(stack.shape[0])
    shift_list = []
    for pi in range(stack.shape[0]):
        shift, _, _ = skimage.registration.phase_cross_correlation(
            stack_for_reg[pi, :, :], fov_for_reg, normalization=None)
        fov_reg = scipy.ndimage.shift(fov, shift)
        fov_reg_stack[pi, :, :] = fov_reg
        if use_valid_pix:
            valid_y, valid_x = np.where(fov_reg > 0)
            corrcoef[pi] = np.corrcoef(stack_pre[pi, valid_y, valid_x].flatten(
            ), fov_reg[valid_y, valid_x].flatten())[0, 1]
        else:
            corrcoef[pi] = np.corrcoef(
                stack_pre[pi, :, :].flatten(), fov_reg.flatten())[0, 1]
        shift_list.append(shift)
    return fov_reg_stack, corrcoef, shift_list


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

