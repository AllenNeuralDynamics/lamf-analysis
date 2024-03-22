import  os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Any, Optional, Union
import json


class BehaviorSessionGrabber(object):
    """Class to get files from an ophys-behavior or behavior only data asset in Code Ocean.

    Data coverage:
    Generally built for LAMF/omFISH style data assets from mesoscope recordings.

    TODO:
    + Add support for session_name
        We could give a  session name or data_asset_id and used the DataAssetLoader class to
        attach the data asset to the capsule. Thefore we dont have to have the asset already attached.
    + Add support for load_options: not sure what this may entail
    + eye_tracking: use DataLoader to automatically load eye tracking if it exits

    Parameters
    ----------
    raw_folder_path : Union[str, Path]
        Path to the raw data folder.
        Attached state: must be already attached.
        Example asset: "multiplane-ophys_639224_2022-10-21_09-48-12"
    eye_tracking_path : Optional[Union[str, Path]]
        Path to the eye tracking data folder.
        Attached state: must be already attached.
        Example asset: "multiplane-ophys_639224_2022-10-21_09-48-12_dlc-eye_2024-03-15_21-23-00"
    session_name: Optional[str]
        The name of the name of session (raw) data_asset. Not implemented yet.
    data_root_path : Optional[str]
        The root path where data assets are stored
    load_options : Optional[dict]
        Options for loading the data asset. Not implemented yet.
    """
    def __init__(self, 
                 raw_folder_path: Union[str, Path] = None,
                 eye_tracking_path: Optional[Union[str, Path]] = None,
                 session_name: Optional[str] = None,
                 data_root_path: Optional[str] = None,
                 load_options: Optional[dict] = None):

        assert raw_folder_path or session_name is not None, \
            "Must provide either raw_folder_path or session_name"

        self.raw_folder_path = Path(raw_folder_path)
        self.eye_tracking_path = Path(eye_tracking_path)
        self.load_options = load_options # not implemented yet
        self.session_name = session_name # not implemented yet

        if data_root_path:
            self.data_root_path = Path(data_root_path)
        else:
            self.data_root_path = Path('../data')

        # processed filepaths dict
        self.file_parts = {"platform_json": "*_platform.json",
                           "eye_mp4": "*[eE]ye*.mp4",
                           "eye_json": "*[eE]ye*.json",
                           "body_mp4": "*[bB]ehavior*.mp4",
                           "body_json": "*[bB]ehavior*.json",
                           "face_mp4": "*[fF]ace*.mp4",
                           "face_json": "*[fF]ace*.json"}
        self._get_file_path_dict(path_to_look=self.raw_folder_path)

        # TODO: store as attb or in file_path_dict
        self.sync_file = self._get_sync_file()
        self.stimulus_pkl = self._get_pkl_file()

        # eye tracking
        eye_file_parts = {"eye_tracking": "*[eE]ye*.csv"}
        eye_file_paths = self._get_file_path_dict(path_to_look=self.eye_tracking_path)
        self.file_parts.update(eye_file_parts)
        self.file_paths.update(eye_file_paths)

    def _find_data_file(self, file_part, path_to_look):
        try:
            file = list(self.path_to_look.glob(f'**/{file_part}'))[0]
        except IndexError:
            file = None
        return file

    def _get_file_path_dict(self,path_to_look):
        file_paths = {}
        for key, value in self.file_parts.items():
            file_paths[key] = self._find_data_file(value,path_to_look)
        self.file_paths = file_paths

    def _get_sync_file(self):
        with open(self.file_paths['platform_json'], 'r') as f:
            platform_json = json.load(f)
        sync_file_path = self.raw_folder_path / "pophys" / platform_json['sync_file']
        self.file_paths['sync_file'] = sync_file_path

    def _get_pkl_file(self):
        with open(self.file_paths['platform_json'], 'r') as f:
            platform_json = json.load(f)

        stimulus_pkl_path = self.raw_folder_path / "pophys" / platform_json['stimulus_pkl']
        self.file_paths["stimulus_pkl"] = stimulus_pkl_path