# This class combines a OphysPlaneDataset and BehaviorDatase to create a BehaviorOphysDataset,
# inputs are raw_folder_path and processed_folder_path
# set at

from data_objects.ophys.ophys_plane_dataset import OphysPlaneDataset
from data_objects.behavior.behavior_dataset import BehaviorDataset


class BehaviorOphysDataset:
    def __init__(self, raw_folder_path, processed_folder_path):
        self.ophys_plane_dataset = OphysPlaneDataset(raw_folder_path, processed_folder_path)
        self.behavior_dataset = BehaviorDataset(raw_folder_path, processed_folder_path)

    def __getattr__(self, name):
        if hasattr(self.ophys_plane_dataset, name):
            return getattr(self.ophys_plane_dataset, name)
        elif hasattr(self.behavior_dataset, name):
            return getattr(self.behavior_dataset, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")