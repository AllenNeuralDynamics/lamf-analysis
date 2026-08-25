from pathlib import Path
import json
import h5py
import pandas as pd

from lamf_analysis.code_ocean import s3_utils
from lamf_analysis.code_ocean import code_ocean_utils as cou
from lamf_analysis.code_ocean import capsule_data_utils as cdu
from lamf_analysis import utils
from lamf_analysis.ophys import zstack


def _is_s3_path(path) -> bool:
    return str(path).startswith('s3://')


def _load_json_by_name(path, filename: str) -> dict:
    """Load a named JSON file from either an S3 prefix or a local directory tree."""
    if _is_s3_path(path):
        files = s3_utils.list_files_from_s3_location(str(path))
        matches = [f for f in files if filename in f]
        if not matches:
            raise FileNotFoundError(f"{filename} not found at {path}")
        return s3_utils.read_json_from_s3(matches[0])
    else:
        candidates = list(Path(path).rglob(filename))
        if not candidates:
            raise FileNotFoundError(f"{filename} not found under {path}")
        with open(candidates[0]) as f:
            return json.load(f)


def get_roi_groups_metadata(path):
    """
    Load roi_groups_metadata.json from an S3 prefix or a local directory.

    Parameters:
        path (str | PathLike): S3 prefix or local directory containing the file.

    Returns:
        dict: Parsed JSON contents of roi_groups_metadata.json.
    """
    return _load_json_by_name(path, 'roi_groups_metadata.json')


def get_scanimage_metadata(path):
    """
    Load scanimage_metadata.json from an S3 prefix or a local directory.

    Parameters:
        path (str | PathLike): S3 prefix or local directory containing the file.

    Returns:
        dict: Parsed JSON contents of scanimage_metadata.json.
    """
    return _load_json_by_name(path, 'scanimage_metadata.json')


def find_stack_xy_info(path):
    """
    Extract XY size and pixel resolution info from ROI groups metadata.

    Parameters:
        path (str | PathLike): S3 prefix or local directory containing roi_groups_metadata.json.

    Returns:
        tuple[list, list]: (sizeXY, pixelResolutionXY) where each is typically a list-like
        structure from the metadata.

    Raises:
        FileNotFoundError: If roi_groups_metadata.json is missing.
        IndexError: If expected keys are not present.
    """
    roi_groups_metadata = get_roi_groups_metadata(path)
    sizeXY = utils.find_keys(roi_groups_metadata, 'sizeXY')[0][1]
    dimXY = utils.find_keys(roi_groups_metadata, 'pixelResolutionXY')[0][1]
    return sizeXY, dimXY


def find_stack_z_info(path):
    """
    Extract Z stack acquisition parameters from ScanImage metadata.

    Parameters:
        path (str | PathLike): S3 prefix or local directory containing scanimage_metadata.json.

    Returns:
        tuple[float, int, int]: (z_step_size_um, num_slices, num_volumes).

    Raises:
        FileNotFoundError: If scanimage_metadata.json is missing.
        IndexError: If expected keys are not present.
        ValueError: If casting to float/int fails.
    """
    scanimage_metadata = get_scanimage_metadata(path)
    z_step_size = float(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualStackZStepSize')[0][1])
    z_num_slices = int(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualNumSlices')[0][1])
    num_volumes = int(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualNumVolumes')[0][1])

    return z_step_size, z_num_slices, num_volumes


def find_stack_acquisition_info(path):
    """
    Determine stack acquisition mode (loop vs step) and actuator/mode settings.

    Parameters:
        path (str | PathLike): S3 prefix or local directory containing scanimage_metadata.json.

    Returns:
        tuple[str, str, str]: (derived_mode, stackActuator, stackMode) where derived_mode is
        'loop' if fastZ + fast mode, else 'step'.

    Raises:
        FileNotFoundError: If scanimage_metadata.json is missing.
        IndexError: If expected keys are not present.
    """
    scanimage_metadata = get_scanimage_metadata(path)
    zstack_actuator = utils.find_keys(scanimage_metadata, 'SI.hStackManager.stackActuator')[0][1]
    zstack_mode = utils.find_keys(scanimage_metadata, 'SI.hStackManager.stackMode')[0][1]
    if (zstack_actuator == "fastZ") and (zstack_mode == "fast"):
        zstack_acquisition_mode = "loop"
    else:
        zstack_acquisition_mode = "step"
    return zstack_acquisition_mode, zstack_actuator, zstack_mode


# Mesoscope calibration: 157 µm per ScanImage sizeXY unit (derived from known pairs
# sizeXY=2.547771 → 400 µm, sizeXY=4.458599 → 700 µm, both giving factor=157.0 exactly).
_SCANIMAGE_UM_PER_SIZE_UNIT = 157.0


def get_xy_size_um(sizeXY):
    """
    Convert a ScanImage sizeXY calibration value to a physical field size in micrometers.

    Uses the mesoscope calibration constant (157 µm / sizeXY unit), which is linear
    through the origin and covers all FOV sizes without a lookup table.

    Parameters:
        sizeXY (float): sizeXY value from roi_groups_metadata.

    Returns:
        float: Physical field size in µm.
    """
    return sizeXY * _SCANIMAGE_UM_PER_SIZE_UNIT


def get_cortical_zstack_reg_df(subject_ids):
    """
    Build a dataframe aggregating cortical z-stack registration derived assets
    and computed spatial resolution metrics.

    Parameters:
        subject_ids (int | str | list[int | str]): One or more subject identifiers.

    Returns:
        pandas.DataFrame: DataFrame with columns:
            s3_path, xy_info, z_info, z_resolution, xy_size_info,
            xy_size_um, xy_size_pix, xy_resolution, plus any original asset columns.

    Notes:
        xy_resolution is computed as xy_size_um / xy_size_pix.
    """
    if isinstance(subject_ids, str) or isinstance(subject_ids, int):
        subject_ids = [subject_ids]
    assert isinstance(subject_ids, list), "type(subject_ids) must be list, str, or int"

    czstack_reg_results_df = pd.DataFrame()
    for subject_id in subject_ids:
        derived_assets_df = cou.get_derived_assets_df(subject_id, 'cortical-zstack-registration',
                                                      add_s3_location=True)
        czstack_reg_results_df = pd.concat([czstack_reg_results_df, derived_assets_df], ignore_index=True)
    czstack_reg_results_df['xy_info'] = czstack_reg_results_df['s3_path'].apply(find_stack_xy_info)
    czstack_reg_results_df['z_info'] = czstack_reg_results_df['s3_path'].apply(find_stack_z_info)

    czstack_reg_results_df['z_resolution'] = czstack_reg_results_df['z_info'].apply(lambda x: x[0])
    czstack_reg_results_df['xy_size_info'] = czstack_reg_results_df['xy_info'].apply(lambda x: x[0][0])
    czstack_reg_results_df['xy_size_um'] = czstack_reg_results_df['xy_size_info'].apply(get_xy_size_um)
    czstack_reg_results_df['xy_size_pix'] = czstack_reg_results_df['xy_info'].apply(lambda x: x[1][0])
    czstack_reg_results_df['xy_resolution'] = czstack_reg_results_df.apply(lambda x: x['xy_size_um'] / x['xy_size_pix'], axis=1)
    return czstack_reg_results_df


########################################################
## Local z-stacks
########################################################


def get_local_zstack_filepath(plane_path):
    plane_path = Path(plane_path)
    local_zstack_path = next(plane_path.rglob(f'*_z_stack_local.h5'), None)
    if local_zstack_path is None:        
        print(f"h5 z-stack file not found in {plane_path}")
        print(f'Looking for h5 z-stack file in the raw path.')
        raw_path = cdu.get_raw_path_from_plane_path(plane_path)
        plane_id = plane_path.name
        local_zstack_paths = list(raw_path.rglob(f'*_z_stack_local.h5'))
        if len(local_zstack_paths) == 0:
            print(f"h5 z-stack file not found in {raw_path}")
            print(f"Looking for tiff z-stack file instead...")
            raw_path = cdu.get_raw_path_from_plane_path(plane_path)
            local_zstack_path = next(raw_path.rglob(f'*_local_z_stack*.tiff'), None)
            if local_zstack_path is None:
                raise FileNotFoundError(f"Z-stack file not found in {plane_path}")
            else:
                return local_zstack_path
        else:
            # find the one with the plane_id in the name
            # if not, use the first one
            local_zstack_path = None
            for p in local_zstack_paths:
                if plane_id in p.name:
                    print(f"Found local z-stack file {p} matching plane_id {plane_id}")
                    local_zstack_path = p
                    break
            if local_zstack_path is None:
                print(f"No local z-stack file with plane_id {plane_id} found.")
                print(f"List of candidate local z-stack files: {local_zstack_paths}")
                print(f"Using the first one: {local_zstack_paths[0]}")
                local_zstack_path = local_zstack_paths[0]
            return local_zstack_path
    else:
        return local_zstack_path


def get_local_zstack_reg_filepath(plane_path):
    plane_path = Path(plane_path)
    local_zstack_reg_path = next(plane_path.rglob(f'*_z_stack_local_reg.h5'), None)
    if local_zstack_reg_path is None:        
        raise FileNotFoundError(f"Z-stack reg file not found in {plane_path}")
    else:
        return local_zstack_reg_path
    

def get_local_zstack_info(plane_path):
    """ Extract z-stack information from a local z-stack HDF5 file.
    Assume local z-stacks are processed on the rig and copied to processed data asset
    """
    local_zstack_path = get_local_zstack_filepath(plane_path)
    if local_zstack_path.suffix == '.tiff':
        _, scanimage_metadata, roi_groups_metadata = \
            zstack.metadata_from_scanimage_tif(local_zstack_path)
    elif local_zstack_path.suffix == '.h5':
        with h5py.File(local_zstack_path, 'r') as f:
            metadata = f['scanimage_metadata'][()]
            metadata = json.loads(metadata)
            roi_groups_metadata = metadata[1]
            scanimage_metadata = metadata[0]
    else:
        raise ValueError(f"Unrecognized local z-stack file format: {local_zstack_path.suffix}")

    sizeXY = utils.find_keys(roi_groups_metadata, 'sizeXY')[0][1][0]
    dimXY = utils.find_keys(roi_groups_metadata, 'pixelResolutionXY')[0][1][0]
    size_xy_um = get_xy_size_um(sizeXY)
    resolution_xy_um = size_xy_um / dimXY

    z_step_size = float(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualStackZStepSize')[0][1])
    z_num_slices = int(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualNumSlices')[0][1])
    num_volumes = int(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualNumVolumes')[0][1])
    zstack_actuator = utils.find_keys(scanimage_metadata, 'SI.hStackManager.stackActuator')[0][1]
    zstack_mode = utils.find_keys(scanimage_metadata, 'SI.hStackManager.stackMode')[0][1]
    if (zstack_actuator == "fastZ") and (zstack_mode == "fast"):
        zstack_acquisition_mode = "loop"
    else:
        zstack_acquisition_mode = "step"

    # make dictionary
    zstack_info = {
        "sizeXY": sizeXY,
        "dimXY": dimXY,
        "size_xy_um": size_xy_um,
        "resolution_xy_um": resolution_xy_um,
        "z_step_size_um": z_step_size,
        "z_num_slices": z_num_slices,
        "num_volumes": num_volumes,
        "zstack_acquisition_mode": zstack_acquisition_mode
    }
    return zstack_info