import json
from pathlib import Path
import time
from typing import List, Dict, Union

import logging
logger = logging.getLogger(__name__)

def make_batch_asset_list_from_table(table,id_key : str = 'raw_asset_id', mount_key: str = 'mount'):
    # note list of lists for batches script
    batch_list = [[{'id': id, 'mount': mount} for id, mount in zip(table[id_key], table[mount_key])]]
    return batch_list

def default_dlc_eye(json_output_path: str, batch_assets_list: list):
    """
    Generate a default settings file for the dlc-eye job.

    Parameters
    ----------
    json_output_path : str
        Path to the output JSON file.
    batch_assets_list : list
        List of lists of assets to include in the job.
        Outer list is batches, inner list is assets in a batch.
        Each batch will be run in parallel.
        Example:
        [
            [{"id": "asset1", "mount": "a1"},
            {"id": "asset2", "mount": "a2"}],
            [{"id": "asset2", "mount": "b1"}]
        ]
    
    TODO: Could make input just list of assets, and look up mount from asset metadata

    """
    settings_dict = {
        "capsule_id": "4cf0be83-2245-4bb1-a55c-a78201b14bfe",
        "tags": ["derived", "eye_tracking", "ophys-mfish"],
        "process_name_suffix": "dlc-eye",
        "assets_list": batch_assets_list
    }

    with open(json_output_path, 'w') as f:
        json.dump(settings_dict, f, indent=4)


def default_cortical_zstack_registration(json_output_path: str, batch_assets_list: list):
    """
    Generate a default settings file for the 

    Parameters
    ----------
    json_output_path : str
        Path to the output JSON file.
    batch_assets_list : list
        List of lists of assets to include in the job.
        Outer list is batches, inner list is assets in a batch.
        Each batch will be run in parallel.
        Example:
        [
            [{"id": "asset1", "mount": "a1"},
            {"id": "asset2", "mount": "a2"}],
            [{"id": "asset2", "mount": "b1"}]
        ]
    
    TODO: Could make input just list of assets, and look up mount from asset metadata

    """
    settings_dict = {
        "capsule_id": "c975fe83-f91d-457e-9e28-596e1e551790",
        "tags": ["derived"],
        "process_name_suffix": "cortical-zstack-reg",
        "assets_list": batch_assets_list
    }

    with open(json_output_path, 'w') as f:
        json.dump(settings_dict, f, indent=4)


def default_cortical_zstack_segmentation(json_output_path: str, batch_assets_list: list):
    """
    Generate a default settings file for the 

    Parameters
    ----------
    json_output_path : str
        Path to the output JSON file.
    batch_assets_list : list
        List of lists of assets to include in the job.
        Outer list is batches, inner list is assets in a batch.
        Each batch will be run in parallel.
        Example:
        [
            [{"id": "asset1", "mount": "a1"},
            {"id": "asset2", "mount": "a2"}],
            [{"id": "asset2", "mount": "b1"}]
        ]
    
    TODO: Could make input just list of assets, and look up mount from asset metadata

    """
    settings_dict = {
        "capsule_id": "0a174d03-4330-4f76-a76c-c56cca4293f0",
        "tags": ["derived"],
        "process_name_suffix": "cortical-zstack-seg",
        "assets_list": batch_assets_list
    }

    with open(json_output_path, 'w') as f:
        json.dump(settings_dict, f, indent=4)


def default_roicat(assets_list: List[Dict],
                  update_params: Dict = {},
                  job_name_suffix: str = time.strftime("%Y%m%d_%H%M%S"),
                  json_output_dir: Union[str, None] = None):
    """
    Generate a default settings dictionary for the session-matching job.
    Optionally save to file if json_output_dir is provided.

    Returns
    -------
    Dict
        The settings dictionary ready to be used with the command line tool
    """
    job_name = "roicat"
    settings_dict = {
        "capsule_id": "71e4e9aa-9b28-4071-b0f2-6dcb9ad74a1e", # MJD roicat
        "tags": ["session-matching", "multiplane-ophys", "derived"],
        "process_name_suffix": "session-matching",
        "assets_list": assets_list,
        "named_parameters": {
            "geometric-method": "RoMa",
            "nonrigid-method": "RoMa",
            "all-to-all": "on",
            "default-fov-scale-factor": None
        }
    }

    # update parameters
    for key, value in update_params.items():
        settings_dict["named_parameters"][key] = value

    settings_json = json.dumps(settings_dict, indent=4)

    # Optionally save to file
    if json_output_dir is not None:
        json_name = f"co_job_{job_name}_{job_name_suffix}.json"
        output_path = Path(json_output_dir) / json_name
        with open(output_path, 'w') as f:
            f.write(settings_json)
        logger.info(f"Settings written to {output_path}")

    return settings_json


def generate_roicat_configs(assets_list: List[Dict],
                            job_name_suffix: str = time.strftime("%Y%m%d_%H%M%S"),
                            json_output_dir: Union[str, None] = None) -> List[Dict]:
    """
    Generate multiple ROICat configurations and return them for direct use.
    Optionally save to files if json_output_dir is provided.
    
    Returns
    -------
    List[Dict]
        List of settings dictionaries ready to be used
    """
    configs = []
    job_name = "roicat"

    # Generate all-to-all ON config
    config_on = default_roicat(
        assets_list=assets_list,
        update_params={"all-to-all": "on",
                       "geometric-method": "PhaseCorrelation",
                       "nonrigid-method": "DeepFlow"}
    )
    configs.append(config_on)

    # Generate all-to-all OFF config
    config_off = default_roicat(
        assets_list=assets_list,
        update_params={"all-to-all": "off",
                       "geometric-method": "PhaseCorrelation",
                       "nonrigid-method": "DeepFlow"}
    )
    configs.append(config_off)

    # Generate all-to-all OFF config
    config_off = default_roicat(
        assets_list=assets_list,
        update_params={"all-to-all": "on",
                       "geometric-method": "DISK_LightGlue",
                       "nonrigid-method": "DeepFlow"}
    )
    configs.append(config_off)


    # Generate all-to-all OFF config
    config_off = default_roicat(
        assets_list=assets_list,
        update_params={"all-to-all": "off",
                       "geometric-method": "DISK_LightGlue",
                       "nonrigid-method": "DeepFlow"}
    )
    configs.append(config_off)

    # put all configs in a single file with "settings_list": [configs]
    if json_output_dir is not None:
        # Simply create a list of the parsed JSON configs
        settings_list = {
            "settings_list": [json.loads(config) for config in configs]
        }
        
        output_path = Path(json_output_dir) / f"co_job_{job_name}_{job_name_suffix}.json"
        with open(output_path, 'w') as f:
            json.dump(settings_list, f, indent=4)
        logger.info(f"All settings written to {output_path}")
    return settings_list