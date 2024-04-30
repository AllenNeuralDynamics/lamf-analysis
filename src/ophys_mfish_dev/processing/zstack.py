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


def plane_paths_from_session(session_path: Union[Path, str],
                             data_level: str = "raw") -> list:
    """Get plane paths from a session directory

    Parameters
    ----------
    session_path : Union[Path, str]
        Path to the session directory
    data_level : str, optional
        Data level, by default "raw". Options: "raw", "processed"

    Returns
    -------
    list
        List of plane paths
    """
    session_path = Path(session_path)
    if data_level == "processed":
        planes = [x for x in session_path.iterdir() if x.is_dir()]
        planes = [x for x in planes if 'nextflow' not in x.name]
    elif data_level == "raw":
        planes = list((session_path / 'ophys').glob('ophys_experiment_*'))
    return planes


def get_motion_correction_crop_xy_range(plane_path: Union[Path, str]) -> tuple:
    """Get x-y ranges to crop motion-correction frame rolling

    # TODO: validate in case where max < 0 or min > 0, which may exist (JK 2023)
    # TODO: use motion_border utils from aind_ophys_utils (04/2024)

    Parameters
    ----------
    plane_path : Path
        Path to the plane directory

    Returns
    -------
    list, list
        Lists of y range and x range, [start, end] pixel index
    """
    motion_csv = list((Path(plane_path) / 'motion_correction').glob(
        '*_motion_transform.csv'))[0]
    motion_df = pd.read_csv(motion_csv)

    max_y = np.ceil(max(motion_df.y.max(), 1)).astype(int)
    min_y = np.floor(min(motion_df.y.min(), 0)).astype(int)
    max_x = np.ceil(max(motion_df.x.max(), 1)).astype(int)
    min_x = np.floor(min(motion_df.x.min(), 0)).astype(int)
    range_y = [-min_y, -max_y]
    range_x = [-min_x, -max_x]
    return range_y, range_x


def local_zstack_metadata(zstack_path: Union[Path, str]) -> tuple:
    """Get scanimage metadata and ROI groups from a local z-stack

    Parameters
    ----------
    zstack_path : Union[Path, str]
        Path to the local z-stack

    Returns
    -------
    dict
        Scanimage metadata
    """
    zstack_path = Path(zstack_path)
    with h5py.File(zstack_path, 'r') as f:
        if 'scanimage_metadata' not in f:
            raise ValueError("scanimage_metadata not found in the h5 file")
        si = f["scanimage_metadata"][()]
    si = si.decode()
    si = json.loads(si)
    scanimage_metadata = si[0]
    roi_groups = si[1]
    return scanimage_metadata, roi_groups


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
    planes = plane_paths_from_session(session_path, data_level="processed")

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
# Local stack
####################################################################################################

def register_local_z_stack(zstack_path):
    """Get registered z-stack, both within and between planes

    Works for step and loop protocol?
    TODO: check if it also works for loop protocol,after fixing the
    rolling effect (JK 2023)

    Parameters
    ----------
    local_z_stack : np.ndarray (3D)
        Local z-stack

    Returns
    -------
    np.ndarray (3D)
        within and between plane registered z-stack
    """
    try:
        # TODO: metadata missing from old files? (04/2024)
        si, roi_groups = local_zstack_metadata(zstack_path)
        number_of_z_planes = si['SI.hStackManager.actualNumSlices']
        number_of_repeats = si['SI.hStackManager.actualNumVolumes']
        # z_step_size = si['SI.hStackManager.actualStackZStepSize']
    except ValueError as e:
        number_of_z_planes = 81
        number_of_repeats = 20
        print(
            f"Error: {e}. Using default values for number_of_z_planes "
            f"({number_of_z_planes}) and number_of_repeats ({number_of_repeats})"
        )

    with h5py.File(zstack_path, 'r') as f:
        local_z_stack = f["data"][()]
    total_num_frames = local_z_stack.shape[0]
    assert total_num_frames == number_of_z_planes * number_of_repeats

    mean_local_zstack_reg = []
    for plane_ind in range(number_of_z_planes):
        single_plane_images = local_z_stack[range(
            plane_ind, total_num_frames, number_of_z_planes), ...]
        single_plane = mean_registered_plane(single_plane_images)
        mean_local_zstack_reg.append(single_plane)

    # Old Scientifica microscope had flyback and ringing in the first 5 frames
    # TODO: reimplement for old rigs (4/2024)
    # if 'CAM2P' in equipment_name:
    #     mean_local_zstack_reg = mean_local_zstack_reg[5:]
    zstack_reg = register_between_planes(np.array(mean_local_zstack_reg))
    return zstack_reg

####################################################################################################
# Z Stack registration
####################################################################################################


def register_between_planes(stack_imgs,
                            ref_ind: int = 30,
                            top_ring_buffer: int = 10,
                            window_size: int = 1,
                            use_adapthisteq: bool = True):
    """Register between planes. Each plane with single 2D image
    Use phase correlation.
    Use median filtered images to calculate shift between neighboring planes.
    Resulting image is not filtered.

    Parameters
    ----------
    stack_imgs : np.ndarray (3D)
        images of a stack. Typically z-stack with each plane registered and averaged.
    ref_ind : int, optional
        index of the reference plane, by default 30
    top_ring_buffer : int, optional
        number of top lines to skip due to ringing noise, by default 10
    window_size : int, optional
        window size for rolling, by default 4
    use_adapthisteq : bool, optional
        whether to use adaptive histogram equalization, by default True

    Returns
    -------
    np.ndarray (3D)
        Stack after plane-to-plane registration.
    """
    num_planes = stack_imgs.shape[0]
    reg_stack_imgs = np.zeros_like(stack_imgs)
    reg_stack_imgs[ref_ind, :, :] = stack_imgs[ref_ind, :, :]
    ref_stack_imgs = med_filt_z_stack(stack_imgs)
    if use_adapthisteq:
        for i in range(num_planes):
            plane_img = ref_stack_imgs[i, :, :]
            ref_stack_imgs[i, :, :] = image_normalization(skimage.exposure.equalize_adapthist(
                plane_img.astype(np.uint16)),dtype='uint16')

    temp_stack_imgs = np.zeros_like(stack_imgs)

    temp_stack_imgs[ref_ind, :, :] = ref_stack_imgs[ref_ind, :, :]
    for i in range(ref_ind + 1, num_planes):
        # Calculation valid pixels
        temp_ref = np.mean(
            temp_stack_imgs[max(0, i - window_size) : i, :, :], axis=0)
        temp_mov = ref_stack_imgs[i, :, :]
        valid_y, valid_x = calculate_valid_pix(temp_ref, temp_mov)

        temp_ref = temp_ref[valid_y[0] +
                            top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]
        temp_mov = temp_mov[valid_y[0] +
                            top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]

        shift, _, _ = skimage.registration.phase_cross_correlation(
            temp_ref, temp_mov, normalization=None, upsample_factor=10)
        temp_stack_imgs[i, :, :] = scipy.ndimage.shift(
            ref_stack_imgs[i, :, :], shift)
        reg_stack_imgs[i, :, :] = scipy.ndimage.shift(
            stack_imgs[i, :, :], shift)
    if ref_ind > 0:
        for i in range(ref_ind - 1, -1, -1):
            temp_ref = np.mean(
                temp_stack_imgs[i + 1 : min(num_planes, i + window_size + 1), :, :], axis=0)
            temp_mov = ref_stack_imgs[i, :, :]
            valid_y, valid_x = calculate_valid_pix(temp_ref, temp_mov)

            temp_ref = temp_ref[valid_y[0] +
                                top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]
            temp_mov = temp_mov[valid_y[0] +
                                top_ring_buffer:valid_y[1] + 1, valid_x[0]:valid_x[1] + 1]

            shift, _, _ = skimage.registration.phase_cross_correlation(
                temp_ref, temp_mov, normalization=None, upsample_factor=10)
            temp_stack_imgs[i, :, :] = scipy.ndimage.shift(
                ref_stack_imgs[i, :, :], shift)
            reg_stack_imgs[i, :, :] = scipy.ndimage.shift(
                stack_imgs[i, :, :], shift)
    return reg_stack_imgs


def mean_registered_plane(images, num_for_ref=None):
    """Get mean FOV of a plane after registration.
    Use phase correlation

    Parameters
    ----------
    images : np.ndarray (3D)
        frames from a plane
    num_for_ref : int, optional
        number of frames to pick for reference, by default None
        When None (or num < 1), then use mean image as reference.

    Returns
    -------
    np.ndarray (2D)
        mean FOV of a plane after registration.
    """
    if num_for_ref is None or num_for_ref < 1:
        ref_img = np.mean(images, axis=0)
    else:
        ref_img, _ = pick_initial_reference(images, num_for_ref)
    reg = np.zeros_like(images)
    for i in range(images.shape[0]):
        shift, _, _ = skimage.registration.phase_cross_correlation(
            ref_img, images[i, :, :], normalization=None)
        reg[i, :, :] = scipy.ndimage.shift(images[i, :, :], shift)
    return np.mean(reg, axis=0)

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

##################################################################################################
# Array Utils
# TODO: move to aind--ophys-utils (04/2024)
##################################################################################################


def calculate_valid_pix(img1, img2, valid_pix_threshold=1e-3):
    """Calculate valid pixels for registration between two images

    Parameters
    ----------
    img1 : np.ndarray (2D)
        Image 1
    img2 : np.ndarray (2D)
        Image 2
    valid_pix_threshold : float, optional
        threshold for valid pixels, by default 1e-3

    Returns
    -------
    list
        valid y range
    list
        valid x range
    """
    y1, x1 = np.where(img1 > valid_pix_threshold)
    y2, x2 = np.where(img2 > valid_pix_threshold)
    # unravel the indices
    valid_y = [max(min(y1), min(y2)), min(max(y1), max(y2))]
    valid_x = [max(min(x1), min(x2)), min(max(x1), max(x2))]
    return valid_y, valid_x


def im_blend(image, overlay, alpha):
    """Blend two images to show match or discrepancy

    Parameters
    ----------
    image : np.ndimage(2d)
        base image
    overlay : np.ndimage(2d)
        image to overlay on top of the base image
    alpha : float ([0,1])
        alpha value for blending

    Returns
    -------
    np.ndimage (2d)
        blended image
    """
    assert len(image.shape) == 2
    assert len(overlay.shape) == 2
    assert image.shape == overlay.shape
    img_uint8 = image_normalization(image, dtype='uint8')
    img_rgb = (np.dstack((img_uint8, np.zeros_like(
        img_uint8), img_uint8))) * (1 - alpha)
    overlay_uint8 = image_normalization(overlay, dtype='uint8')
    overlay_rgb = np.dstack(
        (np.zeros_like(img_uint8), overlay_uint8, overlay_uint8)) * alpha
    blended = img_rgb + overlay_rgb
    return blended


def image_normalization(image, dtype:str='uint16', im_thresh=0):
    """Normalize 2D image and convert to dtype
    Prevent saturation.

    Parameters
    ----------
    image : np.ndarray
        input image (2D)
    dtype : str, optional
        output data type, by default 'uint16'
    im_thresh : float, optional
        threshold when calculating pixel intensity percentile, by default 0
    """
    assert dtype in ['uint8', 'uint16'], "dtype should be either 'uint8' or 'uint16'"

    if dtype == 'uint8':
        dtype = np.uint8
    elif dtype == 'uint16':
        dtype = np.uint16

    clip_image = np.clip(image, np.percentile(
        image[image > im_thresh], 0.2), np.percentile(image[image > im_thresh], 99.8))
    norm_image = (clip_image - np.amin(clip_image)) / \
        (np.amax(clip_image) - np.amin(clip_image)) * 0.9
    norm_image = ((norm_image + 0.05) *
                   np.iinfo(dtype).max * 0.9).astype(dtype)
    return norm_image


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


def rolling_average_stack(stack, n_averaging_planes=5):
    """Rolling average of a z-stack

    Parameters
    ----------
    stack : np.ndarray (3D)
        z-stack to apply rolling average
    n_averaging_planes : int, optional
        number of planes to average, by default 5
        should be in odd number

    Returns
    -------
    np.ndarray (3D)
        rolling average of a z-stack
    """
    stack_rolling = np.zeros_like(stack)
    n_averaging_planes = 5  # should be in odd number
    n_flanking_planes = (n_averaging_planes - 1) // 2
    for i in range(stack.shape[0]):
        if i < n_flanking_planes:
            stack_rolling[i] = np.mean(
                stack[:i + n_flanking_planes + 1], axis=0)
        elif i >= stack.shape[0] - n_flanking_planes:
            stack_rolling[i] = np.mean(stack[i - n_flanking_planes:], axis=0)
        else:
            stack_rolling[i] = np.mean(
                stack[i - n_flanking_planes:i + n_flanking_planes + 1], axis=0)
    return stack_rolling