# This class combines a OphysPlaneDataset and BehaviorDatase to create a BehaviorOphysDataset,
# inputs are raw_folder_path and processed_folder_path
# set at

from data_objects.behavior.behavior_dataset import BehaviorDataset
from data_objects.ophys.ophys_plane_dataset import OphysPlaneDataset


class BehaviorOphysDataset:
    """A class to combine an OphysPlaneDataset and a BehaviorDataset 
    into a single object.

    All attributes of the Other Dataset classes are available as attributes of this class.

    Example #1:
    Assuming the local folders are stuctured like CodeOcean data assets.

    from data_objects.behavior_ophys_dataset import BehaviorOphysDataset
    processed_path = "/allen/programs/mindscope/workgroups/learning/mattd/co_dev/data/1299958728/processed/"
    expt_folder_path = processed_path + "/1299958728"
    raw_path = "/allen/programs/mindscope/workgroups/learning/mattd/co_dev/data/1299958728/raw"
    bod = BehaviorOphysDataset(raw_path, expt_folder_path)

    Example #2:
    Assume raw and processed data assets are attached to capsule.

    """
    def __init__(self, raw_folder_path, expt_folder_path):
        self.ophys_plane_dataset = OphysPlaneDataset(expt_folder_path, raw_folder_path)
        self.behavior_dataset = BehaviorDataset(raw_folder_path)

    def __getattr__(self, name):
        if hasattr(self.ophys_plane_dataset, name):
            return getattr(self.ophys_plane_dataset, name)
        elif hasattr(self.behavior_dataset, name):
            return getattr(self.behavior_dataset, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")