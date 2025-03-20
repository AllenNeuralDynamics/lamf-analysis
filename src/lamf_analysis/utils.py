
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

import ray
os.environ["RAY_verbose_spill_logs"] = "0"

def initialize_ray(spill_dir="/root/capsule/scratch/ray",
                    base_dir='/root/capsule/code'):
    """ Initialize ray """
    sys.path.append(base_dir)
    exclude_files = [str(v.relative_to(Path(base_dir))) for v in Path(base_dir).rglob("*.ipynb")]

    ray.init(ignore_reinit_error=True,
            _temp_dir=spill_dir,
            object_store_memory=(2**10)**3 * 4,
            _system_config={"object_spilling_config": f'{{"type":"filesystem","params":{{"directory_path":"{spill_dir}"}}}}'},
            runtime_env={"working_dir": base_dir,
                        "excludes": exclude_files})


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

    max_shift=512*max_shift_prop
    motion_border = get_max_correction_from_df(motion_df, max_shift=max_shift)
    assert motion_border.down >= 0
    assert motion_border.up >= 0
    assert motion_border.left >= 0
    assert motion_border.right >= 0
    up = 512 if motion_border.up == 0 else -motion_border.up
    right = 512 if motion_border.right == 0 else -motion_border.right

    range_y = [int(motion_border.down), int(up)]
    range_x = [int(motion_border.left), int(right)]

    # max_y = np.ceil(max(motion_df.y.max(), 1)).astype(int)
    # min_y = np.floor(min(motion_df.y.min(), 0)).astype(int)
    # max_x = np.ceil(max(motion_df.x.max(), 1)).astype(int)
    # min_x = np.floor(min(motion_df.x.min(), 0)).astype(int)
    # range_y = [-min_y, -max_y]
    # range_x = [-min_x, -max_x]
    return range_y, range_x


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