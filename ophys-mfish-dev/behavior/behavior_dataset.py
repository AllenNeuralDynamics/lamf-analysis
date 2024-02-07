from ophys.grab_beavior import GrabBehavior

from typing import Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import h5py
import numpy as np
import xarray as xr
from .. import data_processing_keys as keys

# OPHYS_KEYS = ('2p_vsync', 'vsync_2p')

# STIMULUS_KEYS = ('frames', 'stim_vsync', 'vsync_stim')
# PHOTODIODE_KEYS = ('photodiode', 'stim_photodiode')
# EYE_TRACKING_KEYS = ("eye_frame_received",  # Expected eye tracking
#                                             # line label after 3/27/2020
#                         # clocks eye tracking frame pulses (port 0, line 9)
#                         "cam2_exposure",
#                         # previous line label for eye tracking
#                         # (prior to ~ Oct. 2018)
#                         "eyetracking",
#                         "eye_cam_exposing",
#                         "eye_tracking")  # An undocumented, but possible eye tracking line label  # NOQA E114
# BEHAVIOR_TRACKING_KEYS = ("beh_frame_received",  # Expected behavior line label after 3/27/2020  # NOQA E127
#                                                  # clocks behavior tracking frame # NOQA E127
#                                                  # pulses (port 0, line 8)
#                             "cam1_exposure",
#                             "behavior_monitoring")


class LazyLoadable(object):
    def __init__(self, name, calculate):
        ''' Wrapper for attributes intended to be computed or loaded once, 
        then held in memory by a containing object.

        Parameters
        ----------
        name : str
            The name of the hidden attribute in which this attribute's data will be stored.
        calculate : fn
            a function (presumably expensive) used to calculate or load this attribute's data

        '''

        self.name = name
        self.calculate = calculate

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.name):
            setattr(obj, self.name, self.calculate(obj))
        return getattr(obj, self.name)


class BehaviorDataset(GrabBehavior):
    def __init__(self, 
                 raw_folder_path: Optional[str] = None, # where sync file is (pkl file)
                 oeid: Optional[str] = None,
                 data_path: Optional[str] = None):
        super().__init__(raw_folder_path=raw_folder_path,
                         oeid=oeid,
                         data_path=data_path)


    def get_stimulus_presentations(self):
        raw_data_dir = os.path.join(self.data_dir, self.raw_data_folder)
        ophys_session_dir = os.path.join(raw_data_dir, 'pophys')

        # assuming the only .h5 in top level session dir is the sync and the only .pkl is the stim pickle
        pkl_file_path = os.path.join(ophys_session_dir, [file for file in os.listdir(ophys_session_dir) if '.pkl' in file][0])
        pkl_data = pd.read_pickle(pkl_file_path)

        self._stimulus_presentations = stimulus_processing.get_stimulus_presentations(pkl_data, self.stimulus_timestamps)
        return self._stimulus_presentations
    


    stimulus_presentations = LazyLoadable('_stimulus_presentations', get_stimulus_presentations) 