
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