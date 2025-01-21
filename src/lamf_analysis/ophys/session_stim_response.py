import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

class SessionStimResponse:
    def __init__(self, 
                 data_dict=None,
                 session_name=None):
        self.data = {}
        self.session_name = session_name
        if data_dict is not None:
            self.add_data(data_dict)  # Will store (plane, event_type, data_type) -> Dataset
        
    def add_data(self, data_dict):
        """Add data to the SessionStimResponse object"""
        # for (plane, event_type, data_type), df in data_dict.items():
        #     # Convert DataFrame to xarray if needed
        #     if isinstance(df, pd.DataFrame):
        #         self.data[(plane, event_type, data_type)] = self._convert_df_to_xarray(
        #             df, plane, event_type, data_type)
        #     else:
        #         self.data[(plane, event_type, data_type)] = df
        
        self.data.update(data_dict)
        
    def save(self, filepath):
        """Save all data to H5 file"""
        with h5py.File(filepath, 'w') as f:
            f.attrs['session_name'] = self.session_name
            for (plane, event_type, data_type), df in self.data.items():
                # Create nested group path
                group = f.create_group(f"{plane}/{event_type}/{data_type}")
                
                # Save arrays
                group.create_dataset('traces', data=np.stack(df['trace'].values))
                group.create_dataset('timestamps', data=np.stack(df['trace_timestamps'].values))
                
                # Save index columns
                group.create_dataset('stimulus_presentations_id', data=df['stimulus_presentations_id'])
                group.create_dataset('cell_specimen_id', data=df['cell_specimen_id'])
                
                # Save response data
                group.create_dataset('mean_response', data=df['mean_response'])
                group.create_dataset('baseline_response', data=df['baseline_response'])
                group.create_dataset('p_value_gray_screen', data=df['p_value_gray_screen'])
                
                # Save metadata
                group.attrs['ophys_frame_rate'] = df['ophys_frame_rate'].iloc[0]
                group.attrs['interpolate'] = df['interpolate'].iloc[0]
                group.attrs['output_sampling_rate'] = df['output_sampling_rate'].iloc[0]
                group.attrs['response_window_duration'] = df['response_window_duration'].iloc[0]

            
    
    @classmethod
    def load(cls, filepath, plane=None, event_type=None, data_type=None, format='xarray'):
        """Load data from H5 file"""
        obj = cls()
        
        with h5py.File(filepath, 'r') as f:
            if plane and event_type and data_type:
                # Load specific group
                group = f[f"{plane}/{event_type}/{data_type}"]
                if format == 'xarray':
                    obj.data[(plane, event_type, data_type)] = obj._load_group_xarray(
                        group, plane, event_type, data_type)
                elif format == 'pandas':
                    obj.data[(plane, event_type, data_type)] = obj._load_group_pandas(
                        group, plane, event_type, data_type)
            else:
                # Load all groups
                for p in f.keys():
                    for e in f[p].keys():
                        for d in f[p][e].keys():
                            group = f[f"{p}/{e}/{d}"]
                            if format == 'xarray':
                                obj.data[(p, e, d)] = obj._load_group_xarray(
                                    group, p, e, d)
                            elif format == 'pandas':
                                obj.data[(p, e, d)] = obj._load_group_pandas(
                                    group, p, e, d)
        return obj
    
    @classmethod
    def print_contents(cls, filepath):
        """Print contents of H5 file"""
        with h5py.File(filepath, 'r') as f:
            print(f"Session name: {f.attrs['session_name']}")
            for p in f.keys():
                for e in f[p].keys():
                    for d in f[p][e].keys():
                        print(f"Plane: {p}, Event: {e}, Data: {d}")
    
    def _load_group_xarray(self, group, plane, event_type, data_type):
        """Load a single group as xarray Dataset"""
        traces = group['traces'][:]
        timestamps = group['timestamps'][:]
        
        return xr.Dataset(
            data_vars={
                'trace': (['response', 'time'], traces),
                'timestamps': (['response', 'time'], timestamps),
                'mean_response': ('response', group['mean_response'][:]),
                'baseline_response': ('response', group['baseline_response'][:]),
                'p_value_gray_screen': ('response', group['p_value_gray_screen'][:])
            },
            coords={
                'stimulus_presentations_id': ('response', group['stimulus_presentations_id'][:]),
                'cell_specimen_id': ('response', group['cell_specimen_id'][:]),
                'response': np.arange(len(traces)),
                'time': np.arange(traces.shape[1])
            },
            attrs={
                'plane': plane,
                'event_type': event_type,
                'data_type': data_type,
                'ophys_frame_rate': group.attrs['ophys_frame_rate'],
                'interpolate': group.attrs['interpolate'],
                'output_sampling_rate': group.attrs['output_sampling_rate'],
                'response_window_duration': group.attrs['response_window_duration']
            }
        )
        
  
    def _load_group_pandas(self, group, plane, event_type, data_type):
        """Helper function to load a single group as pandas DataFrame"""
        return pd.DataFrame({
            'stimulus_presentations_id': group['stimulus_presentations_id'][:],
            'cell_specimen_id': group['cell_specimen_id'][:],
            'trace': list(group['traces'][:]),
            'trace_timestamps': list(group['timestamps'][:]),
            'mean_response': group['mean_response'][:],
            'baseline_response': group['baseline_response'][:],
            'p_value_gray_screen': group['p_value_gray_screen'][:],
            'ophys_frame_rate': group.attrs['ophys_frame_rate'],
            'plane': plane,
            'data_type': data_type,
            'event_type': event_type,
            'interpolate': group.attrs['interpolate'],
            'output_sampling_rate': group.attrs['output_sampling_rate'],
            'response_window_duration': group.attrs['response_window_duration']
        })
    
    def get_cell_responses(self, cell_id, plane=None, event_type=None, data_type=None):
        """Get all responses for a specific cell"""
        if plane and event_type and data_type:
            ds = self.data[(plane, event_type, data_type)]
            return ds.sel(response=(ds.cell_specimen_id == cell_id))
        else:
            # Return dict of responses across all conditions
            return {key: ds.sel(response=(ds.cell_specimen_id == cell_id))
                   for key, ds in self.data.items()}
    
    def plot_cell_response(self, cell_id, plane, event_type, data_type, 
                          show_individual=True, figsize=(12, 4)):
        """Plot mean response and individual traces for a cell"""
        ds = self.data[(plane, event_type, data_type)]
        cell_data = ds.sel(response=(ds.cell_specimen_id == cell_id))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual traces
        if show_individual:
            for i in range(len(cell_data.response)):
                ax.plot(cell_data.timestamps[i], cell_data.trace[i], 
                       alpha=0.2, color='gray')
        
        # Plot mean ± SEM
        mean_trace = cell_data.trace.mean(dim='response')
        sem_trace = cell_data.trace.std(dim='response') / np.sqrt(len(cell_data.response))
        
        times = cell_data.timestamps[0]
        ax.plot(times, mean_trace, 'b-', label='Mean')
        ax.fill_between(times, 
                       mean_trace - sem_trace,
                       mean_trace + sem_trace,
                       alpha=0.3, color='b')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response')
        ax.set_title(f'Cell {cell_id} responses ({event_type}/{data_type})')
        ax.legend()
        
        return fig, ax
    
    
    def get_concatenated_planes(self, event_type, data_type, format='xarray'):
        """
        Concatenate data from all planes for a specific event_type and data_type.
        
        Args:
            event_type: Type of event to get
            data_type: Type of data to get
            format: 'xarray' or 'pandas' (default: 'xarray')
            
        Returns:
            Combined dataset with plane information preserved
        """
        # Get all matching datasets
        plane_datasets = []
        for (plane, e, d), ds in self.data.items():
            if e == event_type and d == data_type:
                if format == 'xarray':
                    # Add plane as a coordinate
                    ds = ds.assign_coords({'plane': ('response', [plane] * len(ds.response))})
                    plane_datasets.append(ds)
                elif format == 'pandas':
                    df = ds
                    df['plane'] = plane
                    
                    plane_datasets.append(df)
        
        if not plane_datasets:
            raise ValueError(f"No data found for event_type={event_type}, data_type={data_type}")
        
        if format == 'xarray':
            # Concatenate along the response dimension
            return xr.concat(plane_datasets, dim='response')
        elif format == 'pandas':
            # Concatenate DataFrames
            return pd.concat(plane_datasets, axis=0)
        
        
    def plot_planes_comparison(self, event_type, data_type, cell_id=None, stim_id=None, figsize=(8, 4)):
        """
        Plot comparison of responses across planes.
        
        Args:
            event_type: Type of event to plot
            data_type: Type of data to plot
            cell_id: Optional cell ID to filter
            stim_id: Optional stimulus ID to filter
            figsize: Figure size tuple
        """
        ds = self.get_concatenated_planes(event_type, data_type)
        
        # Apply filters
        if cell_id is not None:
            ds = ds.sel(response=(ds.cell_specimen_id == cell_id))
        if stim_id is not None:
            ds = ds.sel(response=(ds.stimulus_presentations_id == stim_id))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot mean ± SEM for each plane
        for plane in np.unique(ds.plane.values):
            plane_data = ds.sel(response=(ds.plane == plane))
            mean_trace = plane_data.trace.mean(dim='response')
            sem_trace = plane_data.trace.std(dim='response') / np.sqrt(len(plane_data.response))
            times = plane_data.timestamps[0]
            
            # using interppolation, upsample mean and sem to 300 length
            mean_trace = np.interp(np.linspace(0, 1, 300), np.linspace(0, 1, len(mean_trace)), mean_trace)
            sem_trace = np.interp(np.linspace(0, 1, 300), np.linspace(0, 1, len(sem_trace)), sem_trace)
            times = np.interp(np.linspace(0, 1, 300), np.linspace(0, 1, len(times)), times)
            
            ax.plot(times, mean_trace, label=f'Plane {plane}')
            ax.fill_between(times, 
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        alpha=0.2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Response')
        title = f'{event_type}/{data_type}'
        if cell_id is not None:
            title += f' - Cell {cell_id}'
        if stim_id is not None:
            title += f' - Stim {stim_id}'
        ax.set_title(title)
        ax.legend()
        
        return fig, ax
    
    
    # class StimResponse:
    # @staticmethod
    # def compute_sem(traces):
    #     """Compute standard error of the mean for a set of traces"""
    #     return np.std(traces, axis=0) / np.sqrt(len(traces))
    
    # @staticmethod
    # def validate_metadata(metadata):
    #     """Check if required metadata fields are present"""
    #     required_fields = ['ophys_frame_rate', 'interpolate', 'output_sampling_rate']
    #     return all(field in metadata for field in required_fields)
    
"""
These would be good static methods because they:
Don't need instance data
Perform general utility functions
Could be used independently: StimResponse.compute_sem(some_traces)
"""
            
        
# Usage examples:
"""
# Load data
sr = StimResponse.load('responses.h5')

# Get concatenated xarray dataset
ds_all = sr.get_concatenated_planes('changes', 'dff', format='xarray')

# Access data with plane information
cell_id = 7
cell_responses = ds_all.sel(response=(ds_all.cell_specimen_id == cell_id))
print(f"Responses from planes: {cell_responses.plane.values}")

# Plot responses colored by plane
plt.figure(figsize=(12, 4))
for plane in np.unique(ds_all.plane.values):
    plane_data = ds_all.sel(response=(ds_all.plane == plane))
    mean_trace = plane_data.trace.mean(dim='response')
    times = plane_data.timestamps[0]
    plt.plot(times, mean_trace, label=f'Plane {plane}')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title(f'Mean responses by plane')
plt.legend()
plt.show()

# Or get as pandas DataFrame
df_all = sr.get_concatenated_planes('changes', 'dff', format='pandas')
print(df_all.groupby('plane').size())  # Count responses per plane
"""