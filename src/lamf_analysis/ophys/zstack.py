import glob
import json
import os
import time
import re
# from multiprocessing import Pool
from dask.distributed import Client
from dask import delayed, compute
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import skimage
from PIL import Image, ImageDraw, ImageFont
from ScanImageTiffReader import ScanImageTiffReader
from tifffile import TiffFile, imread, imwrite
from tqdm import tqdm

####################################################################################################
# Cortical stack
####################################################################################################


def get_zstack_reg(stack, plane_order, n_planes, n_repeats_per_plane, ref_channel, reg_ops):
    """Get registered z-stack, both within and between planes"""

    print(f"Registering zstack for reference channel: {ref_channel}")
    pstring = f"Stack info: plane_order={plane_order}, n_planes={n_planes}," \
              f" n_repeats_per_plane={n_repeats_per_plane}"
    print(pstring)
    new_time = time.time()
    plane_reg_stack, shifts_within = register_within_plane_multi(stack,
                                                                 plane_order=plane_order,
                                                                 n_planes=n_planes,
                                                                 n_repeats_per_plane=n_repeats_per_plane)
    print(f"Frame repeats registered in {np.round(time.time() - new_time, 2)} s")

    print("Registering between planes...")
    new_time = time.time()
    full_reg_stack, shifts_between = reg_between_planes(plane_reg_stack, **reg_ops)
    print(f"Planes registered in {np.round(time.time() - new_time, 2)} s")

    ouput_dict = {'plane_reg_stack': plane_reg_stack,
                  'full_reg_stack': full_reg_stack,
                  'shifts_within': shifts_within,
                  'shifts_between': shifts_between}

    return ouput_dict


def get_zstack_reg_using_shifts(stack, plane_order, n_planes, n_repeats_per_plane,
                                shifts_within, shifts_between,
                                target_channel):
    """Get registered z-stack, both within and between planes"""

    print(f"Registering zstack for channel: {target_channel}, using shifts from reference channel")
    pstring = f"Stack info: plane_order={plane_order}, n_planes={n_planes}," \
              f" n_repeats_per_plane={n_repeats_per_plane}"
    print(pstring)
    new_time = time.time()
    plane_reg_stack, _ = register_within_plane_multi(stack,
                                                     plane_order=plane_order,
                                                     n_planes=n_planes,
                                                     n_repeats_per_plane=n_repeats_per_plane,
                                                     shifts=shifts_within)
    print(f"Frame repeats registered in {np.round(time.time() - new_time, 2)} s")

    print(f"Registering between planes for channel= {target_channel}...")
    new_time = time.time()
    full_reg_stack = reg_between_planes_using_shift_info(plane_reg_stack, shifts_between)
    print(f"Planes registered in {np.round(time.time() - new_time, 2)} s")

    output_dict = {'plane_reg_stack': plane_reg_stack,
                   'full_reg_stack': full_reg_stack,
                   'shifts_within': None,
                   'shifts_between': None}

    return output_dict


def deinterleave_channels(stack: np.ndarray,
                          num_channels: int,
                          ref_channel: int,
                          target_channel: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deinterleave channels from a stack geneated by ScanImage

        Unlcear how channels are sorted in these stacks saved outside
        the typical mesoscope WSE. Sometimes we see 3D stacks, sometimes
        4D.

    Parameters
    ----------
    stack : np.ndarray
        3D or 4D stack
    num_channels : int
        Number of channels in the stack
    ref_channel : int
        Reference channel for registration
    target_channel : int
        Target channel for registration

    Returns
    -------
    np.ndarray
    """
    if len(stack.shape) == 3:  # assume channels interleaved
        stack_ref = stack[ref_channel::num_channels, :, :]
        stack_target = stack[target_channel::num_channels, :, :]
    elif len(stack.shape) == 4:
        stack_ref = np.squeeze(stack[:, ref_channel, :, :])
        stack_target = np.squeeze(stack[:, target_channel, :, :])

    return stack_ref, stack_target


def register_cortical_stack(zstack_path: Union[Path, str],
                            save: bool = False,
                            output_dir: Path = None,
                            zstack_folder: Optional[str] = None,
                            qc_plots: Optional[bool] = False,
                            stack_metadata: Optional[dict] = None,
                            reference_plane: Optional[int] = 60,
                            ref_channel: Optional[int] = None,
                            save_1x_registered: bool = False):
    """Two-step registration of a cortical z-stack up to two channels

    Dev notes
    - 40k frame tiff (1 channel), ~ 8 mins with 12 cores

    Metadata notes:
    - example loop protocol: num_slices=400, num_volumes=100, z_step_size=1, frames_per_slice=1
    - example step protocol: num_slices=100, num_volumes=1, z_step_size=5, frames_per_slice=100
    - z_steps for loop: '[-190 -189 -188 -187 -186 -185 ... n]
    - z_steps for step: '[-190 -190 -190 -190 -190 -190 -185 -185 -185 -185 -185 -185 ... n]
    - Thefore, we can infer if we have a loop or step stack by looking at num_volumes,
        n=1 would be step, n > 1 would be loop.


    Parameters
    ----------
    zstack_path : Union[Path, str]
        Path to tiff stack
    save : bool, optional
        Save registered stacks, by default False
    output_dir : Path, optional
        Path to save registered stacks, by default None
    zstack_folder: str, optional
        Output folder name, by default None
    qc_plots : bool, optional
        Generate QC plots, by default False
    stack_metadata : dict, optional
        Metadata for the stack, in case testing a tiff not generated by ScanImage.
        Keys 'plane_order', 'num_slices', 'num_volumes',
        'num_channels', 'frames_per_slice'.
    reference_plane : int, optional
        Reference plane for between plane registration, by default 60
        Nice to be in the middle of the stack, avoiding top junk
    ref_channel : int, optional
        Reference channel for registration in case of multi-channel stack, by default None
        (within-channel registration)
    save_1x_registered : bool, optional
        Save 1x registered stack, by default False

    """
    start_time = time.time()

    # 0. setup + validate
    zstack_path = Path(zstack_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save and output_dir is None:
        raise ValueError("output_dir must be provided if save is True")

    # output folder (either given name or name of zstack file)
    if zstack_folder is not None:
        output_dir = output_dir / zstack_folder
    else:
        output_dir = output_dir / zstack_path.name.split('.')[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. load stack
    print(f"Loading stack from: {zstack_path}")
    new_time = time.time()

    stack = imread(Path(zstack_path))
    print(f"Stack shape: {stack.shape} read in {np.round(time.time() - new_time, 2)} s")

    # 2. load and parse key metadata
    print("Parsing metadata...")
    new_time = time.time()

    if stack_metadata is None:
        stack_metadata, scanimage_metadata, roi_groups_metadata = metadata_from_scanimage_tif(zstack_path)

        # infer plane_order, see docstring
        if stack_metadata['num_volumes'] == 1:
            plane_order = 'step'
            n_planes = stack_metadata['num_slices']
            n_repeats_per_plane = stack_metadata['frames_per_slice']
        elif stack_metadata['num_volumes'] > 1:
            plane_order = 'loop'
            n_planes = stack_metadata['num_slices']
            n_repeats_per_plane = stack_metadata['num_volumes']
        stack_metadata['plane_order'] = plane_order
    else:
        # in case dict provided
        plane_order = stack_metadata['plane_order']
        n_planes = stack_metadata['num_slices']
        n_repeats_per_plane = stack_metadata['num_volumes']

    print(f"Metadata parsed in {np.round(time.time() - new_time, 2)} s")

    # 3. Register Zstack
    reg_dicts = []  # Main list to store all results
    reg_ops = {'ref_ind': reference_plane, 'top_ring_buffer': 10,
               'window_size': 5, 'use_adapthisteq': True}
    
    
    # num channels; not reliable from scanimage metadata, so just look at dims (05/2024)
    stack_metadata['num_channels'] = 2 if len(stack.shape) == 4 else 1

    # 3A. Single channel
    if stack_metadata['num_channels'] == 1:
        ref_channel = 0  # always 0 for single channel

        reg_dict_ref = get_zstack_reg(stack, plane_order, n_planes,
                                      n_repeats_per_plane, ref_channel,
                                      reg_ops)
        reg_dict_ref['channel'] = ref_channel
        reg_dict_ref['ref_channel'] = ref_channel
        reg_dicts.append(reg_dict_ref)

    # 3B. Two Channel
    elif stack_metadata['num_channels'] == 2:
        has_ref = True
        if ref_channel is None:
            target_channel = 0 # arbitrary assignment
            print(f"Found num_channels = {stack_metadata['num_channels']}, ref_channel = {ref_channel}")
            ref_channel = 1
            has_ref = False
        else:
            target_channel = [i for i in range(stack_metadata['num_channels']) if i != ref_channel][0]
            print(f"Found num_channels = {stack_metadata['num_channels']}, ref_channel = {ref_channel}")

        # reference
        stack_ref, stack_target = deinterleave_channels(stack, stack_metadata['num_channels'],
                                                        ref_channel, target_channel)
        reg_dict_ref = get_zstack_reg(stack_ref, plane_order, n_planes,
                                      n_repeats_per_plane, ref_channel,
                                      reg_ops)
        reg_dict_ref['channel'] = ref_channel
        reg_dict_ref['ref_channel'] = ref_channel
        reg_dicts.append(reg_dict_ref)

        # target
        if has_ref:
            reg_dict_target = get_zstack_reg_using_shifts(stack_target, plane_order, n_planes,
                                                        n_repeats_per_plane,
                                                        reg_dict_ref['shifts_within'],
                                                        reg_dict_ref['shifts_between'],
                                                        target_channel)
            reg_dict_target['channel'] = target_channel
            reg_dict_target['ref_channel'] = ref_channel
        else:
            reg_dict_target = get_zstack_reg(stack_target, plane_order, n_planes,
                                      n_repeats_per_plane, target_channel,
                                      reg_ops)
            reg_dict_target['channel'] = target_channel
            reg_dict_target['ref_channel'] = target_channel
        reg_dicts.append(reg_dict_target)

    # 5. gather processing json
    output_dict = {}
    output_dict.update(stack_metadata)
    output_dict['input_path'] = str(zstack_path)
    output_dict['input_stack_shape'] = stack.shape
    output_dict['reg_ops_between'] = reg_ops
    output_dict['reg_method_within'] = "phase_cross_correlation"
    output_dict['reg_method_between'] = "phase_cross_correlation"
    # channel specific info
    for i, d in enumerate(reg_dicts):
        ch = d['channel']
        output_dict[f'channel_{ch}'] = {'shifts_between': _list_array_to_list(d['shifts_between'])}

    

    # not sure why exists, handling on new stack runs should be outside this function (MJD 01/2025)
    # if processing_fn.exists():
    #     processing_fn_old_list = list(output_dir.glob('registration_processing_*.json'))
    #     all_processing_nums = [int(str(pfn).split('_')[-1].split('.')[0]) for pfn in processing_fn_old_list]
    #     new_processing_num = max(all_processing_nums) + 1
    #     processing_fn = output_dir / f'registration_processing_{new_processing_num:02}.json'

    processing_fn = output_dir / 'registration_processing.json'
    with open(processing_fn, 'w') as f:
        json.dump(output_dict, f, indent=4)

    # save other json
    with open(output_dir / 'scanimage_metadata.json', 'w') as f:
        json.dump(scanimage_metadata, f, indent=4)
    with open(output_dir / 'roi_groups_metadata.json', 'w') as f:
        json.dump(roi_groups_metadata, f, indent=4)

    # 6. save registered stacks + gifs
    if save:
        for i, d in enumerate(reg_dicts):
            ch = d['channel']
            ref_ch = d['ref_channel']
            plane_reg_stack = d['plane_reg_stack']
            full_reg_stack = d['full_reg_stack']

            output_dir_ch = output_dir / f"channel_{ch}_ref_{ref_ch}"
            output_dir_ch.mkdir(parents=True, exist_ok=True)
            
            # fast or slow gif
            if len(plane_reg_stack) > 200:
                duration = 40
            elif len(plane_reg_stack) <= 200:
                duration = 90

            if save_1x_registered:
                reg1_output_path = output_dir_ch / "1x_registered"
                reg1_output_path.mkdir(parents=True, exist_ok=True)
                save_registered_stack(plane_reg_stack, zstack_path, reg1_output_path, n_reg_steps=1)
                save_gif_with_frame_text(plane_reg_stack, zstack_path, reg1_output_path,
                                        n_reg_steps=1, duration=duration, title_str=f'{duration}ms')

            #reg2_output_path = output_dir_ch / "2x_registered"
            reg2_output_path = output_dir_ch
            reg2_output_path.mkdir(parents=True, exist_ok=True)
            save_registered_stack(full_reg_stack, zstack_path, reg2_output_path, n_reg_steps=2)
            save_gif_with_frame_text(full_reg_stack, zstack_path, reg2_output_path,
                                     n_reg_steps=2, duration=duration, title_str=f'{duration}ms')

    # 7. qc_plots
    if qc_plots:
        for i, d in enumerate(reg_dicts):
            ch = d['channel']
            ref_ch = d['ref_channel']
            print("Generating QC figures...")

            if save_1x_registered:
                plane_reg_stack = d['plane_reg_stack']
                reg1_output_path = output_dir / f"channel_{ch}_ref_{ref_ch}/1x_registered"
                qc_figs(plane_reg_stack, zstack_path, reg1_output_path)
            
            full_reg_stack = d['full_reg_stack']
            #reg2_output_path = output_dir / f"channel_{ch}_ref_{ref_ch}/2x_registered"
            reg2_output_path = output_dir / f"channel_{ch}_ref_{ref_ch}"
            qc_figs(full_reg_stack, zstack_path, reg2_output_path)

            print(f"QC figures saved to: {output_dir / 'qc'}")

    print(f"Total time to register cortical stack: {np.round(time.time() - start_time, 2)} s")

    # return plane_reg_stack, full_reg_stack, output_dict
    return output_dict


def metadata_from_scanimage_tif(stack_path):
    """Extract metadata from ScanImage tiff stack

    Dev notes:
    Seems awkward to parse this way
    Depends on ScanImageTiffReader

    Parameters
    ----------
    stack_path : str
        Path to tiff stack

    Returns
    -------
    dict
        stack_metadata: important metadata extracted from scanimage tiff header
    dict
        si_metadata: all scanimge metadata. Each value still a string, so convert if needed.
    dict
        roi_groups_dict: 
    """
    with ScanImageTiffReader(str(stack_path)) as reader:
        md_string = reader.metadata()

    # split si & roi groups, prep for seprate parse
    s = md_string.split("\n{")
    rg_str = "{" + s[1]
    si_str = s[0]

    # parse 1: extract keys and values, dump, then load again
    si_metadata = _extract_dict_from_si_string(si_str)
    # parse 2: json loads works hurray
    roi_groups_dict = json.loads(rg_str)

    stack_metadata = {}
    stack_metadata['num_slices'] = int(si_metadata['SI.hStackManager.actualNumSlices'])
    stack_metadata['num_volumes'] = int(si_metadata['SI.hStackManager.actualNumVolumes'])
    stack_metadata['frames_per_slice'] = int(si_metadata['SI.hStackManager.framesPerSlice'])
    # stack_metadata['z_steps'] = _str_to_int_list(si_metadata['SI.hStackManager.zs'])
    stack_metadata['z_steps'] = _str_to_float_list(si_metadata['SI.hStackManager.zs'])
    stack_metadata['actuator'] = si_metadata['SI.hStackManager.stackActuator']
    # stack_metadata['num_channels'] = sum(_str_to_bool_list(si_metadata['SI.hPmts.powersOn']))
    channels_saved = [ss for ss in re.split('\[|\]| ', si_metadata['SI.hChannels.channelSave']) if len(ss)>0]
    channels_saved = [int(cs) for cs in channels_saved if str(int(cs)) == cs]
    stack_metadata['num_channels'] = len(channels_saved) # TODO: need to check its validity in a larger batch of data
    stack_metadata['channels_saved'] = channels_saved
    # stack_metadata['z_step_size'] = int(si_metadata['SI.hStackManager.actualStackZStepSize'])
    stack_metadata['z_step_size'] = float(si_metadata['SI.hStackManager.actualStackZStepSize'])

    return stack_metadata, si_metadata, roi_groups_dict


####################################################################################################
# Local zstack
####################################################################################################

def _register_stack(stack, total_num_frames, number_of_z_planes):
    mean_local_zstack_reg = []
    for plane_ind in range(number_of_z_planes):
        single_plane_images = stack[range(
            plane_ind, total_num_frames, number_of_z_planes), ...]
        single_plane, shifts = average_reg_plane(single_plane_images)
        mean_local_zstack_reg.append(single_plane)

    # Old Scientifica microscope had flyback and ringing in the first 5 frames
    # TODO: reimplement for old rigs (4/2024)
    # if 'CAM2P' in equipment_name:
    #     mean_local_zstack_reg = mean_local_zstack_reg[5:]
    _zstack_reg, _shifts_between = reg_between_planes(np.array(mean_local_zstack_reg))
    return _zstack_reg


def register_local_zstack_from_raw_tif(zstack_path: Union[Path, str]):
    """ Get registered z-stack, both within and between planes
    From raw tiff stack, meaning that we have to split first
    
    Parameters
    ----------
    local_z_stack : np.ndarray (3D)
        Raw local z-stack, tiff file

    Returns
    -------
    np.ndarray (3D)
        within and between plane registered z-stack
    """
    stack_metadata, _, _ = metadata_from_scanimage_tif(zstack_path)
    num_slices = stack_metadata['num_slices']
    num_volumes = stack_metadata['num_volumes']
    num_channels = stack_metadata['num_channels'] # TODO: need to check its validity in a larger batch of data
    channels_saved = stack_metadata['channels_saved']

    cz_reader = ScanImageTiffReader(str(zstack_path))
    total_num_frames = cz_reader.shape()[0]
    assert total_num_frames == num_slices * num_volumes * num_channels

    try: 
        data = cz_reader.data() # sometimes it fails to read the data (maybe after an update in scanimage?)
    except:
        try: 
            data = imread(zstack_path) # from recent data (12/16/2024) result shape lengths is 4 for multi-channel, instead of 3
        except:
            raise ValueError("Failed to read data from tiff file")
    if num_channels == 1:
        zstack_reg = _register_stack(data, total_num_frames, num_slices)
    elif num_channels > 0:
        zstack_reg = []
        total_num_frames_each_channel = total_num_frames // num_channels
        for ch_ind in range(len(channels_saved)):
            if len(data.shape) == 4:
                ch_split_data = data[:, ch_ind, :, :]
            elif len(data.shape) == 3:
                ch_split_data = data[ch_ind::num_channels, :, :]
            else:
                raise ValueError("Data shape not recognized")
            zstack_reg.append(_register_stack(ch_split_data, 
                              total_num_frames_each_channel, num_slices))
    else:
        raise ValueError("num_channels should be 1 or more")

    return zstack_reg, channels_saved

def decrosstalk_zstack(raw_path, processed_path, opid, paired_opid):
    ''' Decrosstalk a local z-stack using the alpha and beta values from the processing json file

    Parameters
    ----------
    raw_path : Path
        Path to the raw data directory
    processed_path : Path
        Path to the processed data directory
    opid : int
        Ophys plane ID
    paired_opid : int
        Ophys plane ID of the paired plane


    Returns
    -------
    np.ndarray
        Decrosstalked z-stack
    '''
    plane_path = processed_path / str(opid)

    # get local z-stack fn
    # TODO: Use file paths information    
    local_zstack_fn = raw_path / 'pophys' / f'ophys_experiment_{opid}' / f'{opid}_z_stack_local.h5'
    # get paired z-stack fn
    # TODO: Use file paths information
    paired_zstack_fn = raw_path / 'pophys' / f'ophys_experiment_{paired_opid}' / f'{paired_opid}_z_stack_local.h5'

    # Decrosstalk using alpha and beta from the opid
    json_fn = plane_path / 'decrosstalk/processing.json'
    alpha, beta = get_alpha_beta_from_json(json_fn)
    
    with h5py.File(local_zstack_fn, 'r') as f:
        local_zstack = f['data'][:]
    with h5py.File(paired_zstack_fn, 'r') as f:
        paired_zstack = f['data'][:]
    decrosstalked_zstack = np.zeros(local_zstack.shape)
    for zi in range(local_zstack.shape[0]):
        zplane = local_zstack[zi]
        paired_zplane = paired_zstack[zi]
        decrosstalked_plane, _ = apply_mixing_matrix(alpha, beta, zplane, paired_zplane)
        decrosstalked_zstack[zi] = decrosstalked_plane
    return decrosstalked_zstack


# (Potentially) Redundant function from decrosstalk module
def get_alpha_beta_from_json(json_fn):
    with open(json_fn, 'r') as h:
        processing = json.load(h)
    parameters = processing['processing_pipeline']['data_processes'][1]['parameters']
    alpha = parameters['alpha_mean']
    beta = parameters['beta_mean']
    return alpha, beta


# Redundant function from decrosstalk module
def apply_mixing_matrix(alpha, beta, signal_mean, paired_mean):
    """Apply mixing matrix to the mean images to get reconstructed images
    
    Parameters:
    -----------
    alpha : float
        alpha value of the unmixing matrix
    beta : float
        beta value of the unmixing matrix
    signal_mean : np.array
        mean image of the signal plane
    paired_mean : np.array
        mean image of the paired plane

    Returns:
    -----------
    recon_signal : np.array
        reconstructed signal image
    recon_paired : np.array
        reconstructed paired image
    """
    mixing_mat = [[1-alpha, beta], [alpha, 1-beta]]
    unmixing_mat = np.linalg.inv(mixing_mat)
    raw_data = np.vstack([signal_mean.ravel(), paired_mean.ravel()])
    recon_data = np.dot(unmixing_mat, raw_data)
    recon_signal = recon_data[0, :].reshape(signal_mean.shape)
    recon_paired = recon_data[1, :].reshape(paired_mean.shape)
    return recon_signal, recon_paired


def register_local_z_stack(zstack_path, local_z_stack=None):
    """Get registered z-stack, both within and between planes

    Works for step and loop protocol?
    TODO: check if it also works for loop protocol, after fixing the
    rolling effect (JK 2023)

    Parameters
    ----------
    zstack_path : Union[Path, str]
        Path to local z-stack
    local_z_stack : np.ndarray (3D)
        Optional, local z-stack 
        Usually used when registering decrosstalked z-stack

    Returns
    -------
    np.ndarray (3D)
        within and between plane registered z-stack
    """
    try:
        # TODO: metadata missing from old files? (04/2024)
        si_metadata, roi_groups = local_zstack_metadata(zstack_path)
        number_of_z_planes= int(si_metadata['SI.hStackManager.actualNumSlices'])
        number_of_repeats = int(si_metadata['SI.hStackManager.actualNumVolumes'])

        # z_step_size = si['SI.hStackManager.actualStackZStepSize']
    except ValueError as e:
        number_of_z_planes = 81
        number_of_repeats = 20
        print(
            f"Error: {e}. Using default values for number_of_z_planes "
            f"({number_of_z_planes}) and number_of_repeats ({number_of_repeats})"
        )

    if local_z_stack is None:
        with h5py.File(zstack_path, 'r') as f:
            local_z_stack = f["data"][()]
    total_num_frames = local_z_stack.shape[0]
    assert total_num_frames == number_of_z_planes * number_of_repeats

    mean_local_zstack_reg = []
    for plane_ind in range(number_of_z_planes):
        single_plane_images = local_z_stack[range(
            plane_ind, total_num_frames, number_of_z_planes), ...]
        single_plane, shifts = average_reg_plane(single_plane_images)
        mean_local_zstack_reg.append(single_plane)

    # Old Scientifica microscope had flyback and ringing in the first 5 frames
    # TODO: reimplement for old rigs (4/2024)
    # if 'CAM2P' in equipment_name:
    #     mean_local_zstack_reg = mean_local_zstack_reg[5:]
    zstack_reg, shifts_between = reg_between_planes(np.array(mean_local_zstack_reg))
    return zstack_reg


# TODO: remove if not used
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


def image_normalization(image: np.ndarray,
                        dtype: str = 'uint16',
                        im_thresh: float = 0):
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
    norm_image = ((norm_image + 0.05) * np.iinfo(dtype).max * 0.9).astype(dtype)
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


####################################################################################################
# Utils
####################################################################################################


def _list_array_to_list(data):
    if not data:
        return None
    else:
        data = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in data]
    return data


def get_cortical_stack_paths(specimen_folder):
    cz_paths = glob.glob(str(specimen_folder) + "/**/*cortical_z_stack*", recursive=True)

    # session_paths = []
    # for cs in cz_paths:
    #     session_paths.append(Path(cs).parent)
    # session_paths = np.unique(session_paths)

    return cz_paths


def _extract_dict_from_si_string(string):
    """Parse the 'SI' variables from a scanimage metadata string"""

    lines = string.split('\n')
    data_dict = {}
    for line in lines:
        if line.strip():  # Check if the line is not empty
            key, value = line.split(' = ')
            key = key.strip()
            if value.strip() == 'true':
                value = True
            elif value.strip() == 'false':
                value = False
            else:
                value = value.strip().strip("'")  # Remove leading/trailing whitespace and single quotes
            data_dict[key] = value

    json_data = json.dumps(data_dict, indent=2)
    loaded_data_dict = json.loads(json_data)
    return loaded_data_dict


def _str_to_int_list(string):
    return [int(s) for s in string.strip('[]').split()]

def _str_to_float_list(string):
    return [float(s) for s in string.strip('[]').split()]

def _str_to_bool_list(string):
    return [bool(s) for s in string.strip('[]').split()]


def save_registered_stack(reg_stack,
                          zstack_path,
                          output_path,
                          output_fn=None,
                          n_reg_steps=2):
    """Save registered stack as tiff stack

    Parameters
    ----------
    reg_stack : np.array
        Registered stack
    zstack_path : Union[Path, str]
        Path to tiff stack
    output_path : Union[Path, str]
        Path to save registered stack
    n_reg_steps : int, optional
        Number of registration steps, by default 2

    Returns
    -------
    Path
        Path to saved stack
    """
    #zstack_path = Path(zstack_path)

    if n_reg_steps == 1:
        reg_str = "1x"
    elif n_reg_steps == 2:
        reg_str = "2x"
    if output_fn is None:
        #output_fn = zstack_path.stem + '_' + output_path.parent.stem + f'_{reg_str}REG.tif'
        output_fn = output_path.parent.stem + f'_{reg_str}REG.tif'
    save_path = output_path / output_fn

    # for i in range(reg_stack.shape[0]):
    #     imsave(save_path, reg_stack[i], append=True)

    imwrite(save_path, reg_stack)

    return save_path


def load_reg_stack(zstack_path: Union[Path, str],
                   registered_folder: Union[Path, str],
                   n_reg_steps=2):
    """Load registered stack, given original zstack path (for file name)

    Parameters
    ----------
    zstack_path : Union[Path, str]
        Path to tiff stack
    registered_folder : Union[Path, str]
        Path to folder with registered stack
    n_reg_steps : int, optional
        Number of registration steps, by default 2

    Returns
    -------
    np.array
        Registered stack
    """
    zstack_path = Path(zstack_path)
    registered_folder = Path(registered_folder)

    if n_reg_steps == 1:
        reg_str = "1x"
    elif n_reg_steps == 2:
        reg_str = "2x"
    try:
        path = registered_folder / (zstack_path.stem + f'_{reg_str}REG.tif')
        reg_stack = []

        with TiffFile(path) as tif:
            for page in tif.pages:
                reg_stack.append(page.asarray())
        reg_stack = np.array(reg_stack)

        # for page in range(400):
        #     img = imread(path, key=page)
        #     reg_stack.append(img)
        # reg_stack = np.array(reg_stack)
    except FileNotFoundError:
        reg_stack = None
        print(f"Could not find registered stack at: {path}")
    return reg_stack


def normalize_stack_unit8(stack):
    """Normalize stack to 0-255 uint8

    Parameters
    ----------
    stack : np.array
        Stack to normalize

    Returns
    -------
    np.array
        Normalized stack

    TODO: use core utils (aind-ophys-utils) (4/2024)
    """
    norm_stack = []

    for img in stack:
        img_nz = img[img != 0]  # remove all zeros
        p5 = np.percentile(img_nz, 2)
        p98 = np.percentile(img_nz, 99.5)
        img = np.clip(img, p5, p98)
        img = (img - p5) / (p98 - p5)
        img = (img * 255).astype(np.uint8)
        norm_stack.append(img)

    return norm_stack


def save_gif_with_frame_text(reg_stack: np.ndarray,
                             zstack_path: Union[Path, str],
                             output_path: Union[Path, str],
                             n_reg_steps: int = 2,
                             duration: int = 90,
                             title_str: Optional[str] = ''):
    """Save a stack as a gif with n frame in corner of each frame

    Parameters
    ----------
    reg_stack : np.ndarray
        Registered stack
    zstack_path : Union[Path, str]
        Path to tiff stack
    output_path : Union[Path, str]
        Path to save gif
    n_reg_steps : int, optional
        Number of registration steps, by default 2
    duration : int, optional
        Duration of each frame in ms, by default 90
    title_str : str, optional
        Title string for gif, by default ''
    """
    zstack_path = Path(zstack_path)
    output_path = Path(output_path)

    assert n_reg_steps in [1, 2], "n_reg_steps must be 1 or 2"

    if n_reg_steps == 1:
        reg_str = "1x"
    elif n_reg_steps == 2:
        reg_str = "2x"

    fn = output_path / (output_path.parent.stem + f'_{title_str}_{reg_str}REG.gif')

    norm_stack = normalize_stack_unit8(reg_stack)
    frames = []

    frame_text = True
    if frame_text:
        for i in range(reg_stack.shape[0]):
            img = Image.fromarray(norm_stack[i])
            draw = ImageDraw.Draw(img)
            # TODO: find fonts OS-agnostically
            # f = "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf"
            # draw.text((10, 10), f"plane: {i}", fill='yellow', font=ImageFont.truetype(f, 16))
            draw.text((10, 10), f"plane: {i}", fill='yellow')
            frames.append(img)
    else:
        frames = norm_stack
    imageio.mimsave(fn, frames, format='GIF', loop=0, duration=duration)

    return

####################################################################################################
# Registration functions
####################################################################################################


def _reg_single_plane_shift(input):
    """Small wrapper for averge_reg_plane to be used in parallel processing"""
    plane, shifts = input[0], input[1]
    return average_reg_plane_using_shift_info(np.array(plane), shifts)


def _reg_single_plane(frames):
    """Small wrapper for averge_reg_plane to be used in parallel processing"""
    plane_frames_reg, shifts = average_reg_plane(np.array(frames))
    return plane_frames_reg, shifts


def register_within_plane_multi(stack: np.array,
                                plane_order: str,
                                n_planes: int,
                                n_repeats_per_plane: int,
                                shifts: Optional[list] = None,
                                n_processes: Optional[int] = None,
                                cpu_buffer: int = 2):
    """"Register each single plane in a z-stack, uses multiprocessing

    Dev notes:
    - Time ~ 8 mins for 40k frames tif

    Parameters
    ----------
    zstack_path : Union[Path, str]
        Path to tiff stack
    plane_order : str
        Order of planes in stack, either 'step' or 'loop'
        (See docstring for register_cortical_stack for more info)
    n_planes : int, optional
        Number of z_planes in stack
    n_repeats_per_plane : int, optional
        Number of repeats per plane
    shifts : list, optional
        Shifts for each plane, If given will use this for registration
    n_processes : int, optional
        Number of processes to use, by default None
    cpu_buffer : int, optional
        Buffer for number of processes, by default 2

    Returns
    -------
    np.array
        Registered stack
    shifts
        Shifts for each plane
    """

    indices_list = []
    zstack_plane = []
    for i in range(n_planes):
        if plane_order == 'step':
            indices = np.arange(i * n_repeats_per_plane, (i + 1) * n_repeats_per_plane)
        elif plane_order == 'loop':
            indices = np.arange(i, stack.shape[0], n_planes)
        indices_list.append(indices)
        zstack_plane.append(stack[indices])
    indices_list = np.array(indices_list)

    del stack  # save RAM
    n_processes = n_processes if n_processes is not None else os.cpu_count() - cpu_buffer
    if shifts is None:
        # with Pool(n_processes) as p:
        #     result = list(tqdm(p.imap(_reg_single_plane, zstack_plane), total=len(zstack_plane)))
        client = Client()
        tasks = [delayed(_reg_single_plane)(zstack_plane[i]) for i in range(n_planes)]
        results = compute(*tasks, num_workers = n_processes)
        client.close()
        reg_stack = [r[0] for r in results]
        shifts = [r[1] for r in results]
        reg_stack = np.array(reg_stack)
    else:
        input_params = [(zstack_plane[i], shifts[i]) for i in range(len(zstack_plane))]
        # with Pool(n_processes) as p:
            # result = list(tqdm(p.imap(_reg_single_plane_shift, input_params), total=len(input_params)))
        client = Client()
        tasks = [delayed(_reg_single_plane_shift)(input_params[i]) for i in range(n_planes)]
        results = compute(*tasks, num_workers = n_processes)
        client.close()
        
        reg_stack = np.array(results)
    return reg_stack, shifts


def reg_between_planes(stack_imgs,
                       ref_ind: int = 30,
                       top_ring_buffer: int = 10,
                       window_size: int = 5,
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
        window size for rolling, by default 5
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
            timg = skimage.exposure.equalize_adapthist(plane_img.astype(np.uint16))
            ref_stack_imgs[i, :, :] = image_normalization(timg, dtype='uint16')

    temp_stack_imgs = np.zeros_like(stack_imgs)

    temp_stack_imgs[ref_ind, :, :] = ref_stack_imgs[ref_ind, :, :]
    shift_all = []
    shift_all.append([0, 0])
    for i in range(ref_ind + 1, num_planes):
        # Calculation valid pixels
        temp_ref = np.mean(
            temp_stack_imgs[max(0, i - window_size):i, :, :], axis=0)
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
        shift_all.append(shift)
    if ref_ind > 0:
        for i in range(ref_ind - 1, -1, -1):
            temp_ref = np.mean(
                temp_stack_imgs[i + 1: min(num_planes, i + window_size + 1), :, :], axis=0)
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
            shift_all.insert(0, shift)
    return reg_stack_imgs, shift_all


def average_reg_plane(images: np.ndarray) -> Union[np.ndarray, list]:
    """Get mean FOV of a plane after registration.
    Use phase correlation

    Parameters
    ----------
    images : np.ndarray (3D)
        frames from a plane

    Returns
    -------
    np.ndarray (2D)
        mean FOV of a plane after registration.
    """

    # if num_for_ref is None or num_for_ref < 1:
    #   ref_img = np.mean(images, axis=0)
    ref_img, _ = pick_initial_reference(images)
    reg = np.zeros_like(images)
    shift_all = []
    for i in range(images.shape[0]):
        shift, _, _ = skimage.registration.phase_cross_correlation(
            ref_img, images[i, :, :], normalization=None)
        reg[i, :, :] = scipy.ndimage.shift(images[i, :, :], shift)
        shift_all.append(shift)
    return np.mean(reg, axis=0), shift_all


def average_reg_plane_using_shift_info(images, shift_all):
    """Get mean FOV of a plane after registration using pre-calculated shifts.
    Resulting image is not filtered.

    Parameters
    ----------
    images : np.ndarray (3D)
        frames from a plane
    shift_all : list
        list of shifts between neighboring frames.
        The length should be the same as the number of frames of images (shape[0]).

    Returns
    -------
    np.ndarray (2D)
        mean FOV of a plane after registration.
    """
    num_planes = images.shape[0]
    assert len(shift_all) == num_planes
    reg = np.zeros_like(images)
    for i in range(num_planes):
        reg[i, :, :] = scipy.ndimage.shift(images[i, :, :], shift_all[i])
    return np.mean(reg, axis=0)


def reg_between_planes_using_shift_info(stack_imgs, shift_all):
    """Register between planes using pre-calculated shifts.
    Each plane with single 2D image.
    Resulting image is not filtered.

    Parameters
    ----------
    stack_imgs : np.ndarray (3D)
        images of a stack. Typically z-stack with each plane registered and averaged.
    shift_all : list
        list of shifts between neighboring planes.
        The length should be the same as the number of planes of stack_images (shape[0]).

    Returns
    -------
    np.ndarray (3D)
        Stack after plane-to-plane registration.
    """
    stack_imgs = np.array(stack_imgs)
    num_planes = stack_imgs.shape[0]
    assert len(shift_all) == num_planes
    reg_stack_imgs = np.zeros_like(stack_imgs)
    for i in range(num_planes):
        reg_stack_imgs[i, :, :] = scipy.ndimage.shift(
            stack_imgs[i, :, :], shift_all[i])
    return reg_stack_imgs


def pick_initial_reference(frames: np.ndarray, num_for_ref: int = 20) -> np.ndarray:
    """ computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    From suite2p.registration.register

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    nimg, Ly, Lx = frames.shape
    frames = np.reshape(frames, (nimg, -1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = np.matmul(frames, frames.T)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis=1)
    bestCC = np.mean(CCsort[:, 1:num_for_ref], axis=1)
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    selected_frame_inds = indsort[0:num_for_ref]
    refImg = np.mean(frames[selected_frame_inds, :], axis=0)
    refImg = np.reshape(refImg, (Ly, Lx))
    return refImg, selected_frame_inds


####################################################################################################
# Plot functions
####################################################################################################


def plot_xz(stack: np.array,
            agg_func: str = 'max',
            y_slice: tuple = None,
            title_info: str = '',
            clahe: bool = True,
            ax: plt.Axes = None,
            colorbar: bool = False):
    """Plot the projection of a stack in the XZ plane. Use agg_func to
        aggregate the Z dimension.

    Parameters
    ----------
    stack : np.array
        Stack to plot, ZYX
    agg_func : str, optional
        Aggregation function to use, by default 'max'. Options: 'mean', 'max', 'top95'
    y_slice : tuple, optional
        Slice to plot, by default None
    title_info : str, optional
        Additional info for title, by default ''
    clahe : bool, optional
        Apply CLAHE normalization, by default True
    ax : Optional[plt.Axes], optional
        Matplotlib axis, by default None
    colorbar : bool, optional
        Add colorbar, by default False

    Returns
    -------
    fig
        Matplotlib figure
    """

    sns.set_context('notebook')

    yslice_str = "all" if y_slice is None else f"{y_slice[0]}-{y_slice[1]}"

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 10))
    else:
        fig = ax.figure

    if y_slice is None:
        y_slice = slice(None)
    else:
        y_slice = slice(y_slice[0], y_slice[1])

    img = stack.copy()
    img = img[:, y_slice, :]

    if agg_func == 'mean':
        img = img.mean(axis=1)
    elif agg_func == 'max':
        img = img.max(axis=1)
    elif agg_func == 'top95':
        img = np.percentile(img, 95, axis=1)

    if clahe:
        img = skimage.exposure.equalize_adapthist(img.astype(np.uint16))
    img = image_normalization(img, dtype='uint16')

    vmax = np.percentile(img, 99.9)
    ax.set_aspect('equal')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Z (plane)')

    im = ax.imshow(img, vmax=vmax)
    # color bar
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.3)
    title = f'XZ {agg_func} proj - y={yslice_str} \n {title_info}'
    ax.set_title(title)

    return fig


def plot_xy(stack: np.array, stack_path, z_slice=None, ax=None):

    sns.set_context('notebook')

    stack_path = Path(stack_path)
    session_id = stack_path.stem.split('_')[0]
    zslice_str = "all" if z_slice is None else f"{z_slice[0]}-{z_slice[1]}"

    if z_slice is None:
        z_slice = slice(None)
    else:
        z_slice = slice(z_slice[0], z_slice[1])

    img = stack.copy()
    img = img[z_slice, :, :]
    img = img.max(axis=0)
    img = skimage.exposure.equalize_adapthist(img.astype(np.uint16))
    img = image_normalization(img, dtype='uint16')

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 10))
    else:
        fig = ax.figure
    vmax = np.percentile(img, 99.9)

    # square ax
    ax.set_aspect('equal')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.imshow(img, vmax=vmax)
    title = f'XY max proj - osid={session_id} - z_slice={zslice_str}'
    ax.set_title(title)

    return fig


def plot_plane_intensity(stack: np.array,
                         agg_func: str = 'max',
                         ax: Optional[plt.Axes] = None):
    """Plot the average intensity of each z plane in a stack

    Parameters
    ----------
    stack : np.array
        Stack to plot
    agg_func : str, optional
        Aggregation function to use, by default 'mean'. Options: 'mean', 'max', 'top95'
    ax : Optional[plt.Axes], optional
        Matplotlib axis, by default None

    Returns
    -------
    fig
        Matplotlib figure
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5))
    else:
        fig = ax.figure

    if agg_func == 'mean':
        int = stack.mean(axis=1)
    elif agg_func == 'max':
        int = stack.max(axis=1)
    elif agg_func == 'top95':
        int = np.percentile(stack, 95, axis=1)

    int = int.mean(axis=1)
    ax.plot(int)
    ax.set_title(f'Mean of {agg_func} across z planes')
    ax.set_xlabel('Z plane')
    ax.set_ylabel('Avg pix intensity')

    return fig


####################################################################################################
# QC & Fig functions
####################################################################################################

def save_fig(fig, output_path, dpi=300):
    """Save a figure to a path

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure
    output_path : Union[Path, str]
        Path to save figure
    dpi : int, optional
        DPI, by default 300
    """
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi)

    return


def fig_4_xy_projections(stack: np.ndarray,
                         zstack_path: Path):
    """Figure with 4 subplots showing xy projections (top down) of stack
    top left: all planes
    top right: top 5 planes
    bottom left: mid 5 planes
    bottom right: bottom 5 planes


    Parameters
    ----------
    stack : np.array
        3D stack of registered planes
    zstack_path : Path
        Path to original zstack tiff file

    Returns
    -------
    fig : plt.Figure
    """

    n_planes = stack.shape[0]

    # get midplane
    mid = n_planes // 2

    top_slice = (1, 5)
    mid_slice = (mid - 2, mid + 2)
    bottom_slice = (n_planes - 5, n_planes - 1)

    sns.set_context('notebook')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_xy(stack, zstack_path, z_slice=None, ax=axs[0, 0])
    plot_xy(stack, zstack_path, z_slice=top_slice, ax=axs[0, 1])
    plot_xy(stack, zstack_path, z_slice=mid_slice, ax=axs[1, 0])
    plot_xy(stack, zstack_path, z_slice=bottom_slice, ax=axs[1, 1])

    plt.tight_layout()

    return fig


def fig_4_xz_projections(stack: np.ndarray,
                         zstack_path: Path):
    """Figure with 4 subplots showing xz projections of stack
    top left: all planes
    top right: top 5 planes
    bottom left: mid 5 planes
    bottom right: bottom 5 planes


    Parameters
    ----------
    stack : np.array
        3D stack of registered planes
    zstack_path : Path
        Path to original zstack tiff file

    Returns
    -------
    fig : plt.Figure
    """

    n_planes = stack.shape[2]
    mid = n_planes // 2
    first_slice = (1, 20)
    mid_slice = (mid - 10, mid + 10)
    last_slice = (n_planes - 20, n_planes - 1)
    sns.set_context('notebook')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plot_xz(stack, y_slice=None, clahe=False, ax=axs[0, 0])
    plot_xz(stack, y_slice=first_slice, clahe=False, ax=axs[0, 1])
    plot_xz(stack, y_slice=mid_slice, clahe=False, ax=axs[1, 0])
    plot_xz(stack, y_slice=last_slice, clahe=False, ax=axs[1, 1])

    plt.tight_layout()

    return fig


def fig_plane_intensity(stack: np.ndarray,
                        zstack_path: Path):
    """Figure showing mean intensity of each plane in stack

    Parameters
    ----------
    stack : np.array
        3D stack of registered planes
    zstack_path : Path
        Path to original zstack tiff file

    Returns
    -------
    fig : plt.Figure
    """

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    session_id = Path(zstack_path).stem.split('_')[0]
    ts = f"osid: {session_id}"

    plot_plane_intensity(stack, agg_func="max", ax=axs[0])
    plot_plane_intensity(stack, agg_func="mean", ax=axs[1])
    plot_plane_intensity(stack, agg_func="top95", ax=axs[2])

    plot_xz(stack, agg_func="max", title_info=ts, clahe=False, y_slice=None, ax=axs[3])
    plot_xz(stack, agg_func="mean", title_info=ts, clahe=False, y_slice=None, ax=axs[4])
    plot_xz(stack, agg_func="top95", title_info=ts, clahe=False, y_slice=None, ax=axs[5])

    plt.tight_layout()
    return fig


def qc_figs(stack: np.ndarray,
            zstack_path: Union[Path, str],
            output_folder: Union[Path, str]):
    """Generate all qc figures for a registered stack

    Parameters
    ----------
    stack : np.array
        3D stack of registered planes
    zstack_path : Path
        Path to original zstack tiff file

    Returns
    -------
    fig : plt.Figure
    """

    zstack_path = Path(zstack_path)
    output_folder = Path(output_folder) / 'qc'
    output_folder.mkdir(parents=True, exist_ok=True)

    session_id = Path(zstack_path).stem.split('_')[0]
    ts = f"osid: {session_id}"

    # plot xy
    fig1 = fig_4_xy_projections(stack, zstack_path)
    fig2 = fig_4_xz_projections(stack, zstack_path)
    fig3 = plot_xz(stack, title_info=ts, y_slice=None)
    fig4 = plot_xy(stack, zstack_path, z_slice=None)
    fig5 = fig_plane_intensity(stack, zstack_path)

    save_dict = [{'figure': fig1, 'name': 'xy_projection_slices'},
                 {'figure': fig2, 'name': 'xz_projection_slice'},
                 {'figure': fig3, 'name': 'xz_projection_all'},
                 {'figure': fig4, 'name': 'xy_projection_all'},
                 {'figure': fig5, 'name': 'pixel_intensity_across_z'}]

    for d in save_dict:
        save_fig(d['figure'], output_folder / f"{d['name']}.png")
