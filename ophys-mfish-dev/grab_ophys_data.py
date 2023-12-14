import  os
import numpy as np
import pandas as pd
from pathlib import Path


class GrabOphysOutputs(object):
    def __init__(self, expt_folder_path):
        self.expt_folder_path = Path(expt_folder_path)
        self.oeid = self.expt_folder_path.stem

    def print_expt_folder_path(self):
        print(self.expt_folder_path)


    
