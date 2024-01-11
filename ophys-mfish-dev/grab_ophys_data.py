import  os
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from typing import Optional


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

    def _find_expt_folder_from_oeid(self, oeid):
        # find in results
        found = list(self.data_path.glob(f'**/{oeid}'))
        assert found != 1, f"Found {len(found)} folders with oeid {oeid}"
        return found[0]

    def _find_data_file(self, file_part):
        # find in expt_folder_path
        file = list(self.expt_folder_path.glob(f'**/*{file_part}'))[0]
        return file

    ####
    # Data files
    ####

    def get_motion_correction_crop_xy_range(self):
        file_part = "motion_transform.csv"
        file = self._find_data_file(file_part)
        return file
