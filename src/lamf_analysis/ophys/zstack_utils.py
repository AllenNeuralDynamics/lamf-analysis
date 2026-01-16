from pathlib import Path
import json
import h5py
import pandas as pd

from lamf_analysis.code_ocean import s3_utils
from lamf_analysis.code_ocean import code_ocean_utils as cou
from lamf_analysis import utils


def get_roi_groups_metadata(s3_path):
    """
    Load roi_groups_metadata.json from an S3 path.

    Parameters:
        s3_path (str | PathLike): S3 prefix containing the metadata file.

    Returns:
        dict: Parsed JSON contents of roi_groups_metadata.json.

    Raises:
        FileNotFoundError: If the file is not found in the provided S3 path.
    """
    files = s3_utils.list_files_from_s3_location(str(s3_path))
    roi_groups_metadata_fn_list = [f for f in files if 'roi_groups_metadata.json' in f]
    if len(roi_groups_metadata_fn_list) == 0:
        raise FileNotFoundError("No roi_groups_metadata.json file found.")
    else:
        roi_groups_metadata_fn = roi_groups_metadata_fn_list[0]    
        roi_groups_metadata = s3_utils.read_json_from_s3(roi_groups_metadata_fn)
    return roi_groups_metadata


def get_scanimage_metadata(s3_path):
    """
    Load scanimage_metadata.json from an S3 path.

    Parameters:
        s3_path (str | PathLike): S3 prefix containing the metadata file.

    Returns:
        dict: Parsed JSON contents of scanimage_metadata.json.

    Raises:
        FileNotFoundError: If the file is not found in the provided S3 path.
    """
    files = s3_utils.list_files_from_s3_location(str(s3_path))
    scanimage_metadata_fn_list = [f for f in files if 'scanimage_metadata.json' in f]
    if len(scanimage_metadata_fn_list) == 0:
        raise FileNotFoundError("No scanimage_metadata.json file found.")
    else:
        scanimage_metadata_fn = scanimage_metadata_fn_list[0]    
        scanimage_metadata = s3_utils.read_json_from_s3(scanimage_metadata_fn)
    return scanimage_metadata


def find_stack_xy_info(s3_path):
    """
    Extract XY size and pixel resolution info from ROI groups metadata.

    Parameters:
        s3_path (str | PathLike): S3 prefix containing roi_groups_metadata.json.

    Returns:
        tuple[list, list]: (sizeXY, pixelResolutionXY) where each is typically a list-like
        structure from the metadata.

    Raises:
        FileNotFoundError: If roi_groups_metadata.json is missing.
        IndexError: If expected keys are not present.
    """
    roi_groups_metadata = get_roi_groups_metadata(s3_path)
    sizeXY = utils.find_keys(roi_groups_metadata, 'sizeXY')[0][1]
    dimXY = utils.find_keys(roi_groups_metadata, 'pixelResolutionXY')[0][1]
    return sizeXY, dimXY


def find_stack_z_info(s3_path):
    """
    Extract Z stack acquisition parameters from ScanImage metadata.

    Parameters:
        s3_path (str | PathLike): S3 prefix containing scanimage_metadata.json.

    Returns:
        tuple[float, int, int]: (z_step_size_um, num_slices, num_volumes).

    Raises:
        FileNotFoundError: If scanimage_metadata.json is missing.
        IndexError: If expected keys are not present.
        ValueError: If casting to float/int fails.
    """
    scanimage_metadata = get_scanimage_metadata(s3_path)
    z_step_size = float(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualStackZStepSize')[0][1])
    z_num_slices = int(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualNumSlices')[0][1])
    num_volumes = int(utils.find_keys(scanimage_metadata, 'SI.hStackManager.actualNumVolumes')[0][1])
    
    return z_step_size, z_num_slices, num_volumes


def find_stack_acquisition_info(s3_path):
    """
    Determine stack acquisition mode (loop vs step) and actuator/mode settings.

    Parameters:
        s3_path (str | PathLike): S3 prefix containing scanimage_metadata.json.

    Returns:
        tuple[str, str, str]: (derived_mode, stackActuator, stackMode) where derived_mode is
        'loop' if fastZ + fast mode, else 'step'.

    Raises:
        FileNotFoundError: If scanimage_metadata.json is missing.
        IndexError: If expected keys are not present.
    """
    scanimage_metadata = get_scanimage_metadata(s3_path)
    zstack_actuator = utils.find_keys(scanimage_metadata, 'SI.hStackManager.stackActuator')[0][1]
    zstack_mode = utils.find_keys(scanimage_metadata, 'SI.hStackManager.stackMode')[0][1]
    if (zstack_actuator == "fastZ") and (zstack_mode == "fast"):
        zstack_acquisition_mode = "loop"
    else:
        zstack_acquisition_mode = "step"
    return zstack_acquisition_mode, zstack_actuator, zstack_mode


def get_xy_size_um(sizeXY, tol=0.005):
    """
    Map a raw sizeXY calibration value to a physical field size in micrometers.

    Parameters:
        sizeXY (float): Calibration value from metadata.
        tol (float): Tolerance for matching known calibration values.

    Returns:
        int | None: Physical size (um) if matched (e.g., 400 or 700), else None.
    """
    if abs(sizeXY - 2.547771) < tol:
        return 400
    elif abs(sizeXY - 4.458599) < tol:
        return 700
    else:
        raise ValueError(f"Unrecognized sizeXY calibration value: {sizeXY}")


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
        derived_assets_df = cou.get_derived_assets_df(subject_id, 'cortical-zstack-registration')
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
    plane_name = plane_path.stem    
    local_zstack_path = plane_path / f'{plane_name}_z_stack_local.h5'
    if not local_zstack_path.exists():        
        raise FileNotFoundError(f"Z-stack file not found: {local_zstack_path}")
    else:
        return local_zstack_path


def get_local_zstack_reg_filepath(plane_path):
    plane_path = Path(plane_path)
    plane_name = plane_path.stem    
    local_zstack_path = plane_path / f'movie_qc/{plane_name}_z_stack_local_reg.h5'
    if not local_zstack_path.exists():        
        raise FileNotFoundError(f"Z-stack file not found: {local_zstack_path}")
    else:
        return local_zstack_path
    

def get_local_zstack_info(plane_path):
    """ Extract z-stack information from a local z-stack HDF5 file.
    Assume local z-stacks are processed on the rig and copied to processed data asset
    """
    local_zstack_path = get_local_zstack_filepath(plane_path)    
    with h5py.File(local_zstack_path, 'r') as f:
        metadata = f['scanimage_metadata'][()]
        metadata = json.loads(metadata)
        roi_groups_metadata = metadata[1]
        scanimage_metadata = metadata[0]

    sizeXY = _find_keys(roi_groups_metadata, 'sizeXY')[0][1][0]
    dimXY = _find_keys(roi_groups_metadata, 'pixelResolutionXY')[0][1][0]
    size_xy_um = get_xy_size_um(sizeXY, dimXY)
    resolution_xy_um = size_xy_um / dimXY

    z_step_size = float(_find_keys(scanimage_metadata, 'SI.hStackManager.actualStackZStepSize')[0][1])
    z_num_slices = int(_find_keys(scanimage_metadata, 'SI.hStackManager.actualNumSlices')[0][1])
    num_volumes = int(_find_keys(scanimage_metadata, 'SI.hStackManager.actualNumVolumes')[0][1])
    zstack_actuator = _find_keys(scanimage_metadata, 'SI.hStackManager.stackActuator')[0][1]
    zstack_mode = _find_keys(scanimage_metadata, 'SI.hStackManager.stackMode')[0][1]
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