from pathlib import Path
import os
import sys
import h5py
import numpy as np
import json
from typing import Union
import skimage
import scipy
import pandas as pd
import cv2
from aind_ophys_utils.motion_border_utils import get_max_correction_from_df


####################################################################################################
# Code Ocean: Ophys
####################################################################################################

def check_ophys_folder(path):
    ophys_names = ['ophys', 'pophys', 'mpophys']
    ophys_folder = None
    for ophys_name in ophys_names:
        ophys_folder = path / ophys_name
        if ophys_folder.exists():
            break
        else:
            ophys_folder = None

    return ophys_folder


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
        planes = [x for x in planes if ('nextflow' not in x.name) and ('nwb' not in x.name)]
    elif data_level == "raw":
        raw_ophys_folder_name_bases = ['ophys', 'pophys', 'mpophys']
        for raw_ophys_folder_name_base in raw_ophys_folder_name_bases:
            raw_ophys_folder = session_path / raw_ophys_folder_name_base
            if raw_ophys_folder.exists():
                break
            else:
                raw_ophys_folder = None
        if raw_ophys_folder is not None:
            planes = [x for x in raw_ophys_folder.iterdir() if x.is_dir()] # could be none for those uploaded directly from rig
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
    try:
        processing_json_fn = list((Path(plane_path) / 'motion_correction').glob(
            'processing.json'))[0]
        processing_json = json.load(open(processing_json_fn))
        max_shift_prop = processing_json['processing_pipeline']['data_processes'][0]['parameters']['suite2p_args']['maxregshift']
    except:
        processing_json_fn = list((Path(plane_path) / 'motion_correction').glob(
            '*_motion_correction_data_process.json'))[0]
        processing_json = json.load(open(processing_json_fn))
        max_shift_prop = processing_json['parameters']['suite2p_args']['maxregshift']
    
    motion_csv = list((Path(plane_path) / 'motion_correction').glob(
        '*_motion_transform.csv'))[0]
    motion_df = pd.read_csv(motion_csv)

    session_json = get_session_json_from_plane_path(plane_path)
    fov_info = session_json['data_streams'][0]['ophys_fovs'][0] # assume this data is the same for all fovs
    fov_height = fov_info['fov_height']
    fov_width = fov_info['fov_width']

    max_shift = max(fov_height, fov_width) * max_shift_prop
    motion_border = get_max_correction_from_df(motion_df, max_shift=max_shift)
    assert motion_border.down >= 0
    assert motion_border.up >= 0
    assert motion_border.left >= 0
    assert motion_border.right >= 0
    up = fov_height if motion_border.up == 0 else fov_height - motion_border.up
    right = fov_width if motion_border.right == 0 else fov_width - motion_border.right

    range_y = [int(motion_border.down), int(up)]
    range_x = [int(motion_border.left), int(right)]

    return range_y, range_x


def get_session_json_from_plane_path(plane_path):
    ''' Load session.json for a given plane path
    # Note: capsule_data_utils.py has the same function, but cannot import cdu due to circular import
    '''
    if isinstance(plane_path, str):
        plane_path = Path(plane_path)
    if not os.path.isdir(plane_path):
        raise ValueError(f'Path not found ({plane_path})')
    except StopIteration:
        session_name = plane_path.parent.name.split('_processed')[0]
        raw_path = plane_path.parent.parent / session_name
        session_json_fn = raw_path / 'session.json'
    with open(session_json_fn) as f:
        session_json = json.load(f)
    return session_json


####################################################################################################
## Mean response
####################################################################################################
def condition_rename(mean_response_df, condition_version):
    conditions = mean_response_df.condition.unique()
    if condition_version == 1:
        pass # not implemented yet
    elif condition_version == 2:
        condition_map = {}
        for condition in conditions:
            if 'image_name in [' in condition:
                condition_map[condition] = 'all-images'
            elif 'image_name==' in condition:
                temp_image_name = condition.split('==')[1].split(' ')[0].strip('"')
                if 'flashes_since_change' in condition:
                    condition_map[condition] = temp_image_name
                elif 'is_change' in condition and 'hit' not in condition and 'miss' not in condition:
                    condition_map[condition] = f'change - {temp_image_name}'
                elif 'is_change and hit' in condition:
                    condition_map[condition] = f'hit - {temp_image_name}'
                elif 'is_change and miss' in condition:
                    condition_map[condition] = f'miss - {temp_image_name}'
                else:
                    raise ValueError(f'Unknown condition: {condition}')
            elif condition == 'omitted':
                condition_map[condition] = 'omission'
            elif condition == 'is_change':
                condition_map[condition] = 'change'
            elif condition == 'is_change and hit':
                condition_map[condition] = 'hit'
            elif condition == 'is_change and miss':
                condition_map[condition] = 'miss'
            else:
                raise ValueError(f'Unknown condition: {condition}')
        mean_response_df['condition_query_str'] = mean_response_df.condition
        mean_response_df['condition'] = mean_response_df.condition.map(condition_map)
    elif condition_version == 3:
        pass # not implemented yet
    else:
        raise ValueError(f'Invalid condition_version: {condition_version}')