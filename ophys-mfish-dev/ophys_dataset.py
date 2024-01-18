from grab_ophys_outputs import GrabOphysOutputs
from typing import Any, Optional
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import h5py
import numpy as np


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


class OphysDataset(GrabOphysOutputs):
    def __init__(self, 
                 expt_folder_path: Optional[str] = None,
                 oeid: Optional[str] = None,
                 data_path: Optional[str] = None):
        super().__init__(expt_folder_path=expt_folder_path,
                         oeid=oeid,
                         data_path=data_path)

    ####################################################################
    # Data files
    ####################################################################
        
    def _add_csid_to_table(self, table):
        """Cell specimen ids are not avaiable in CodeOcean, as they were in LIMS (01/18/2024)
        Use this method to add them.
        
        Option 1: duplicated cell_roi_id
        Currently, cell_roi_ids are just indexes. Eventually they will be given numbers as well.
        """

        # Option 1: just duplicated cell_roi_id
        # check table index name
        if table.index.name == 'cell_roi_id':
            table['cell_specimen_id'] = table.index.values
        elif table.columns.contains('cell_roi_id'):
            table['cell_specimen_id'] = table.cell_roi_id
        else:
            raise Exception('Table does not contain cell_roi_id')
        table = table.set_index('cell_specimen_id')

        return table

    def get_average_projection_png(self):
        self._average_projection = plt.imread(self.file_paths['average_projection_png'])
        return self._average_projection
    average_projection = LazyLoadable('_average_projection', get_average_projection_png)

    def get_max_projection_png(self):
        self._max_projection = plt.imread(self.file_paths['max_projection_png'])
        return self._max_projection
    max_projection = LazyLoadable('_max_projection', get_max_projection_png)

    def get_motion_transform_csv(self):
        self._motion_transform = pd.read_csv()
        return self._motion_transform
    motion_transform = LazyLoadable('_motion_transform', get_motion_transform_csv)

    # TODO: should we rename the attribute to segmentation?
    def get_cell_specimen_table(self): 
        with open(self.file_paths['segmentation_output_json']) as json_file:
            segmentation_output = json.load(json_file)
        cell_specimen_table = pd.DataFrame(segmentation_output)
        cell_specimen_table = cell_specimen_table.rename(columns={'id': 'cell_roi_id'})
        print(cell_specimen_table.columns)
        cell_specimen_table = self._add_csid_to_table(cell_specimen_table)
        self._cell_specimen_table = cell_specimen_table
        return self._cell_specimen_table
    cell_specimen_table = LazyLoadable('_cell_specimen_table', get_cell_specimen_table)

    def get_raw_fluorescence_traces(self):

        with h5py.File(self.file_paths['roi_traces_h5'], 'r') as f:
            traces = np.asarray(f['data'])
            roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]

        traces_df = pd.DataFrame(index=roi_ids, columns=['raw_fluorescence_traces'])
        for i, roi_id in enumerate(roi_ids):
            traces_df.loc[roi_id, 'raw_fluorescence_traces'] = traces[i, :]
        traces_df = traces_df.rename(columns={'roi_id': 'cell_roi_id'})
        traces_df = self._add_csid_to_table(traces_df)
        self._raw_fluorescence_traces = traces_df
        return self._raw_fluorescence_traces
    raw_fluorescence_traces = LazyLoadable('_raw_fluorescence_traces', get_raw_fluorescence_traces)

    def get_neuropil_traces(self):
        # TODO: cell_roi_ids are removed from this table. Should we add them back?
        # TODO: shoudl we rename this attribute to neuropil_corrected_traces?

        f = h5py.File(self.file_paths['neuropil_correction_h5'], mode='r')
        neuropil_traces_array = np.asarray(f['FC'])
        roi_ids = [int(roi_id) for roi_id in np.asarray(f['roi_names'])]
        RMSE = [value for value in np.asarray(f['RMSE'])]
        r = [value for value in np.asarray(f['r'])]

        # convert to dataframe 
        neuropil_traces = pd.DataFrame(index=roi_ids, columns=['neuropil_fluorescence_traces', 'r', 'RMSE'])
        for i, roi_id in enumerate(roi_ids):
            neuropil_traces.loc[roi_id, 'neuropil_fluorescence_traces'] = neuropil_traces_array[i, :]
            neuropil_traces.loc[roi_id, 'r'] = r[i]
            neuropil_traces.loc[roi_id, 'RMSE'] = RMSE[i]
        neuropil_traces.index.name = 'cell_roi_id'
        neuropil_traces = self._add_csid_to_table(neuropil_traces)
        self._neuropil_traces = neuropil_traces
        return self._neuropil_traces
    neuropil_traces = LazyLoadable('_neuropil_traces', get_neuropil_traces) 

    @classmethod
    def construct_and_load(cls, experiment_id, cache_dir=None, **kwargs):
        ''' Instantiate a VisualBehaviorOphysDataset and load its data

        Parameters
        ----------
        experiment_id : int
            identifier for this experiment
        cache_dir : str
            directory containing this experiment's

        '''

        obj = cls(experiment_id, cache_dir=cache_dir, **kwargs)

        obj.get_max_projection_png()
        obj.get_average_projection_png()
        obj.get_motion_transform_csv()

        # obj.get_metadata()
        # obj.get_timestamps()
        # obj.get_ophys_timestamps()
        # obj.get_stimulus_timestamps()
        # obj.get_behavior_timestamps()
        # obj.get_eye_tracking_timestamps()
        # obj.get_stimulus_presentations()
        # obj.get_stimulus_template()
        # obj.get_stimulus_metadata()
        # obj.get_running_speed()
        # obj.get_licks()
        # obj.get_rewards()
        # obj.get_task_parameters()
        # obj.get_trials()
        # obj.get_dff_traces_array()
        # obj.get_corrected_fluorescence_traces()
        # obj.get_events_array()
        # obj.get_cell_specimen_table()
        # obj.get_roi_mask_dict()
        # obj.get_roi_mask_array()
        # obj.get_cell_specimen_ids()
        # obj.get_cell_indices()
        # obj.get_dff_traces()
        # obj.get_events()
        # obj.get_pupil_area()
        # obj.get_extended_stimulus_presentations()

        return obj