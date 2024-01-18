import  os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Any, Optional


class GrabOphysOutputs(object):
    def __init__(self, 
                 expt_folder_path: Optional[str] = None,
                 oeid: Optional[str] = None,
                 data_path: Optional[str] = None):

        assert expt_folder_path or oeid is not None, "Must provide either expt_folder_path or oeid"

        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = Path('../data')

        if oeid:
            self.oeid = oeid
            self.expt_folder_path = self._find_expt_folder_from_oeid(oeid)
        elif expt_folder_path:
            self.expt_folder_path = Path(expt_folder_path)
            self.oeid = self.expt_folder_path.stem

        # filepaths dict
        self.file_parts = {"platform_json": "_platform.json",
                           "processing_json": "processing.json",
                           "params_json": "_params.json",
                           "registered_metrics_json": "_registered_metrics.json",
                           "output_json": "_output.json",
                           "average_projection_png": "_average_projection.png",
                           "max_projection_png": "_max_projection.png",
                           "motion_transform_csv": "_motion_transform.csv",
                           "segmentation_output_json": "segmentation_output.json",
                           "roi_traces_h5": "roi_traces.h5",
                           "neuropil_correction_h5": "neuropil_correction.h5",
                           "neuropil_masks_json": "neuropil_masks.json",
                           "neuropil_trace_output_json": "neuropil_trace_output.json",
                           "demixing_output_h5": "demixing_output.h5",
                           "demixing_output_json": "demixing_output.json",
                           "dff_h5": "dff.h5",
                           "extract_traces_json": "extract_traces.json",
                           "events_oasis_h5": "events_oasis.h5"}
        self.file_paths = {}
        self._get_file_path_dict()

    def _find_expt_folder_from_oeid(self, oeid):
        # find in results
        found = list(self.data_path.glob(f'**/{oeid}'))
        assert found != 1, f"Found {len(found)} folders with oeid {oeid}"
        return found[0]

    def _find_data_file(self, file_part):
        # find in expt_folder_path
        try:
            file = list(self.expt_folder_path.glob(f'**/*{file_part}'))[0]
        except IndexError:
            file = None
        return file

    ####################################################################
    # Data files
    ###################################################################

    def _get_file_path_dict(self):
        for key, value in self.file_parts.items():
            self.file_paths[key] = self._find_data_file(value)
        return self.file_paths