import os
import numpy as np
import pandas as pd
import warnings
import json
import glob
from pathlib import Path
import h5py
import time
import fnmatch

from codeocean import CodeOcean
from codeocean.data_asset import (DataAssetSearchParams,
                                  DataAssetAttachParams)
from codeocean.components import SearchFilter

import aind_session
from aind_session import Session
from aind_ophys_data_access import capsule
from comb.behavior_ophys_dataset import BehaviorOphysDataset, BehaviorMultiplaneOphysDataset
from aind_ophys_data_access import rois
from comb import file_handling

from lamf_analysis.code_ocean import capsule_bod_utils as cbu
import lamf_analysis.utils as lamf_utils
from lamf_analysis.code_ocean import (docdb_utils,
                                      s3_utils)
from lamf_analysis.code_ocean import code_ocean_utils as cou

import logging
logger = logging.getLogger(__name__)

DEFAULT_MOUNT_TO_IGNORE = ['fb4b5cef-4505-4145-b8bd-e41d6863d7a9', # Ophys_Extension_schema_10_14_2024_13_44
                            '35d1284e-4dfa-4ac3-9ba8-5ea1ae2fdaeb'], # ROI classifier V1
TIME_FORMAT = '[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
DATE_FORMAT = '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'


def get_mouse_session_df(subject_id,
                         processed_date_after=None,
                         processed_date_before=None,
                         include_pupil=True):
    ''' Get mouse session dataframe.
    Using aind_session, it first querys all sessions for a mouse (raw and derived data all matched to the raw session),
        potentially filters by processed_date_after and processed_date_before,
        and then filters by include_pupil.
    Resulting dataframe has the following columns:
        raw_data_date, processed_data_date, capsule_id, commit_id, processed_data_asset_id, raw_data_asset_id, num_provenence_data_assets, pupil_data_asset_id
        * capsule_id and commit_id can be used to filter (hopefully using look-up table)
    '''
    success = True
    # mouse_sessions = aind_session.get_sessions(subject_id=subject_id) # This errors out with "unauthorized issue"
    # Temporary fix while awaiting SciComp solution 2025/09/30 JK
    mouse_sessions = cou.get_mouse_sessions_by_filters(subject_id=subject_id)
    # to prevent errors (happens when adding faulty tags)
    mouse_sessions = tuple([ms for ms in mouse_sessions if ms.subject_id == str(subject_id)])

    raw_data_date_list = []
    processed_data_date_list = []
    capsule_ids_list = []
    commit_ids_list = []
    processed_data_asset_ids_list = []
    raw_data_asset_ids_list = []
    
    num_provenence_data_assets_list = []
    if include_pupil:
        pupil_data_asset_ids_list = []
    for session in mouse_sessions:
        try: 
            data_name = session.raw_data_asset.name
        except:
            continue
        if 'multiplane-ophys' not in data_name:
            continue
        raw_date = session.raw_data_asset.name.split('_')[2] 
        processed_data = [da for da in session.data_assets if '_processed_' in da.name]
        # processed_data = [da for da in processed_data if (da.provenance.commit is not None)]
        if include_pupil:
            pupil_data = [da for da in session.data_assets if 'dlc-eye' in da.name]
            pupil_raw_data = [np.setdiff1d(da.provenance.data_assets, DEFAULT_MOUNT_TO_IGNORE) for da in pupil_data]

        processed_data_dates = [da.name.split('_processed_')[1].split('_')[0] for da in processed_data]
        capsule_ids = [da.provenance.capsule for da in processed_data]
        commit_ids = [da.provenance.commit for da in processed_data]
        data_asset_ids = [da.id for da in processed_data]
        raw_data_asset_ids = [np.setdiff1d(da.provenance.data_assets, DEFAULT_MOUNT_TO_IGNORE) for da in processed_data]
        num_provenence_data_assets = [len(da.provenance.data_assets) for da in processed_data]
        for i in range(len(processed_data)):
            if processed_date_after is not None:
                if processed_data_dates[i] < processed_date_after:
                    continue
            if processed_date_before is not None:
                if processed_data_dates[i] > processed_date_before:
                    continue

            raw_data_asset_id = raw_data_asset_ids[i]

            if include_pupil:
                matching_pupil_data_ind = np.where([raw_data_asset_id in pupil_raw_data[j] for j in range(len(pupil_raw_data))])[0]
                if len(matching_pupil_data_ind) == 1:
                    pupil_data_asset_ids_list.append(pupil_data[matching_pupil_data_ind[0]].id)
                elif len(matching_pupil_data_ind) == 0:
                    pupil_data_asset_ids_list.append(0)
                else:
                    raise ValueError(f'More than one matching pupil data asset found for {raw_data_asset_id} from {session}')

            raw_data_date_list.append(raw_date)
            capsule_ids_list.append(capsule_ids[i])
            commit_ids_list.append(commit_ids[i])
            processed_data_asset_ids_list.append(data_asset_ids[i])
            processed_data_date_list.append(processed_data_dates[i])
            raw_data_asset_ids_list.append(raw_data_asset_id)
            num_provenence_data_assets_list.append(num_provenence_data_assets[i])
    mouse_session_df = pd.DataFrame({'raw_data_date': raw_data_date_list,
                                        'processed_data_date': processed_data_date_list,
                                        'capsule_id': capsule_ids_list,
                                        'commit_id': commit_ids_list,
                                        'processed_data_asset_id': processed_data_asset_ids_list,
                                        'raw_data_asset_id': raw_data_asset_ids_list,
                                        'num_provenence_data_assets': num_provenence_data_assets_list})
    if include_pupil:
        mouse_session_df['pupil_data_asset_id'] = pupil_data_asset_ids_list

    mouse_session_df['num_raw_data_asset_ids'] = mouse_session_df['raw_data_asset_id'].apply(len)
    if np.all(mouse_session_df['num_raw_data_asset_ids'].values == 1):
        mouse_session_df['raw_data_asset_id'] = mouse_session_df['raw_data_asset_id'].apply(lambda x: x[0])
    else:
        success = False
        warnings.warn('Multiple raw data asset ids found for a single processed data asset id')
    
    if include_pupil:
        if np.any(mouse_session_df['pupil_data_asset_id'].values == 0):
            success = False
            warnings.warn(f'No matching pupil data asset found for {mouse_session_df[mouse_session_df["pupil_data_asset_id"] == 0].raw_data_date.values}')
        
    return success, mouse_session_df


def add_dff_long_baseline_window_to_mouse_df(mouse_df):
    processed_asset_ids = mouse_df.processed_data_asset_id.to_list()
    dff_long_windows = docdb_utils.get_dff_long_baseline_window(processed_asset_ids)
    dff_long_window_dict = {item['code_ocean_id']: item['long_window'] for item in dff_long_windows}
    mouse_df['dff_long_window'] = mouse_df['processed_data_asset_id'].map(dff_long_window_dict)
    return mouse_df


def get_cortical_zstack_sessions(subject_id,
                                 czstack_name_regex='*_cortical_z_stack*.tif*',
                                 czstack_key='column_z_stack',
                                 verbose=False):
    ''' Get all cortical zstack sessions for a given subject_id (subject_id)
    '''
    session_infos = docdb_utils.get_session_infos_from_docdb(subject_id=subject_id)
    def _find_keys(d, key_substr):
        keys = []
        for k, v in d.items():
            if key_substr in k:
                keys.append((k, v))
            if isinstance(v, dict):
                keys.extend(_find_keys(v, key_substr))
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        keys.extend(_find_keys(item, key_substr))
        return keys
    
    # check sessions with cortical z-stack from s3 file names
    czstack_sessions = []
    czstack_fn_list = []
    num_error = 0
    for session_info in session_infos.itertuples():
        filepaths = s3_utils.list_files_from_s3_location(session_info.s3_path)
        # find filepaths that match with czstack_name_regex (within s3 filepaths)
 
        czstack_fn = fnmatch.filter(filepaths, czstack_name_regex)
        assert len(czstack_fn) < 2, f"More than one czstack found in {session_info.raw_asset_name} with regex {czstack_name_regex}"
        if len(czstack_fn) == 0:
            if verbose:
                print(f"No czstack with the input format {czstack_name_regex}")
                print("Checking platform.json")
 
            platform_json_s3_path = fnmatch.filter(filepaths, '*platform.json')
            if len(platform_json_s3_path) != 1:
                if verbose:
                    print('='*40)
                    print('                 ERROR')
                    print('='*40)
                    print(f"No platform.json found in {session_info.raw_asset_name}, skipping session.")
                num_error += 1
                continue
            platform_json_s3_path = platform_json_s3_path[0]
            platform_json = s3_utils.read_json_from_s3(platform_json_s3_path)
            czstack_fn_tuple = _find_keys(platform_json, czstack_key)
 
            if not czstack_fn_tuple:
                if verbose:
                    print(f"No column_z_stack found in {platform_json_s3_path}, skipping session.")
                continue
            if len(set(czstack_fn_tuple)) > 1:
                if verbose:
                    print('='*40)
                    print('                 ERROR')
                    print('='*40)
                    print(f"Multiple column_z_stack keys found in {platform_json_s3_path}")
                num_error += 1
                continue
            czstack_fn_candidate = czstack_fn_tuple[0][1]
            czstack_fn = fnmatch.filter(filepaths, czstack_fn_candidate)
            if len(czstack_fn) == 0:
                if verbose:
                    print('='*40)
                    print('                 ERROR')
                    print('='*40)
                    print(f"column_z_stack file {czstack_fn_candidate} from platform.json not found in {session_info.raw_asset_name}, skipping session.")
                num_error += 1
                continue
            elif len(czstack_fn) > 1:
                if verbose:
                    print('='*40)
                    print('                 ERROR')
                    print('='*40)
                    print(f"Multiple files found for column_z_stack file {czstack_fn_candidate} from platform.json in {session_info.raw_asset_name}, skipping session.")
                num_error += 1
                continue
        czstack_fn = czstack_fn[0].split('/')[-1]
        czstack_fn_list.append(czstack_fn)
        czstack_sessions.append(session_info)
    if num_error > 0:
        print(f"\nContinuing despite {num_error} errors so far.")
    else:
        print("\nNo errors found.")
    czstack_sessions = pd.DataFrame(czstack_sessions)
    return czstack_sessions, czstack_fn_list
        

def attach_mouse_data_assets(mouse_session_df, include_pupil=True):
    ''' Attach mouse data assets to mouse session dataframe.
    Built to use the results from get_mouse_session_df.
    Returns if successful.
    '''
    assert np.all([isinstance(raw_id, str) for raw_id in mouse_session_df.raw_data_asset_id.values]), \
        'raw data asset ids must be str'
    if include_pupil:
        assert np.all([isinstance(raw_id, str) for raw_id in mouse_session_df.raw_data_asset_id.values]), \
        f'"include_pupil" set to {include_pupil}, so must provide appropriate pupil data asset ids'
    success = True
    try:
        capsule.attach_assets(mouse_session_df.raw_data_asset_id.values)
        capsule.attach_assets(mouse_session_df.processed_data_asset_id.values)
        if include_pupil:
            capsule.attach_assets(mouse_session_df.pupil_data_asset_id.values)
    except:
        success = False
    return success


def attach_assets(asset_ids:list, client=None):
    """Attach list of asset_ids to capusle with CodeOcean SDK, print mount state
    
    Parameters
    ----------
    assets : list
        list of asset_ids
        Example: ['1az0c240-1a9z-192b-pa4c-22bac5ffa17b', '1az0c240-1a9z-192b-pa4c-22bac5ffa17b']
    client : object
        CodeOcean client object
        If None, must set "API_SECRET" in environment variable for CodeOcean token
        
    Returns
    -------
    None
    """
    
    if client is None:
        client = cou.get_co_client()

    # DataAssetAttachParams(id="1az0c240-1a9z-192b-pa4c-22bac5ffa17b", mount="Reference")
    data_assets = [DataAssetAttachParams(id=aid) for aid in asset_ids]        
            
    results = client.capsules.attach_data_assets(
        capsule_id=os.getenv("CO_CAPSULE_ID"),
        attach_params=data_assets,
    )

    for target_id in asset_ids:
        result = next((item for item in results if item.id == target_id), None)

        if result:
            ms = result.mount_state
            logger.info(f"asset_id: {target_id} - mount_state: {ms}")
            print(f"asset_id: {target_id} - mount_state: {ms}")
        else:
            print(f"asset_id: {target_id} - not found in CodeOcean API response")
    return


def check_attached_data_assets(mouse_df_to_attach, 
                               data_dir=Path('/root/capsule/data'),
                               include_pupil=False, pupil_suffix='dlc-eye'):
    expected_num_assets = 3 if include_pupil else 2
    raw_asset_names = mouse_df_to_attach['raw_asset_name'].values
    for _, row in mouse_df_to_attach.iterrows():
        raw_asset_name = row['raw_asset_name']
        processed_date = row['processed_data_date']        
        matching_dirs = [dd for dd in list(data_dir.iterdir()) if (raw_asset_name in dd.name)]
        if len(matching_dirs) == 0:
            print(f'No matching data dir found for raw asset {raw_asset_name}')
            return False
        if len(matching_dirs) > expected_num_assets:
            print(f'More than expected data dirs found for raw asset {raw_asset_name}: {matching_dirs}')
            return False
        raw_dir = [dd for dd in matching_dirs if dd.name == raw_asset_name]
        if len(raw_dir) == 0:
            print(f'No raw data dir found for raw asset {raw_asset_name}')
            return False
        processed_dir = [dd for dd in matching_dirs if processed_date in dd.name]
        if len(processed_dir) == 0:
            print(f'No processed data dir found for raw asset {raw_asset_name} with processed date {processed_date}')
            return False
        elif len(processed_dir) > 1:
            print(f'More than one processed data dir found for raw asset {raw_asset_name} with processed date {processed_date}: {processed_dir}')
            return False
        if include_pupil:
            pupil_dir = [dd for dd in matching_dirs if pupil_suffix in dd.name]
            if len(pupil_dir) == 0:
                print(f'No pupil data dir found for raw asset {raw_asset_name}')
                return False
            elif len(pupil_dir) > 1:
                print(f'More than one pupil data dir found for raw asset {raw_asset_name}: {pupil_dir}')
                return False
    return True
    

def attach_data_assets_with_repeats(mouse_df_to_attach, data_dir=Path('/root/capsule/data'), 
                                    max_repeats=5, include_pupil=False, pupil_suffix='dlc-eye'):
    for repeat_i in range(max_repeats):
        attach_success = attach_mouse_data_assets(mouse_df_to_attach,
                                                  include_pupil=include_pupil)
        if attach_success:
            print(f'It says attach success after {repeat_i+1} tries!')
            check_success = check_attached_data_assets(mouse_df_to_attach,
                                                      data_dir=data_dir,
                                                      include_pupil=include_pupil,
                                                      pupil_suffix=pupil_suffix)
            if check_success:
                print(f'Attach verified after {repeat_i+1} tries!')
                return True
            else:
                print(f'Attach verification failed after {repeat_i+1} tries. Retrying...')
                time.sleep(2)
        else:
            print(f'Attach failed on try {repeat_i+1}. Retrying...')
    print(f'Attach failed after {max_repeats} tries.')
    return False


def get_session_info(subject_id, data_dir='/root/capsule/data'):
    ''' Get all raw data paths in the data directory
    '''
    data_folders = [d for d in glob.glob(data_dir + '/*') if Path(d).is_dir()]
    raw_paths = np.sort([d for d in data_folders if ('processed' not in d.split('/')[-1]) and
                        ('dlc-eye' not in d.split('/')[-1]) and
                        ('multiplane-ophys' in d.split('/')[-1]) and 
                        ('stimuli' not in d.split('/')[-1]) and
                        ('stim-response' not in d.split('/')[-1]) and
                        ('ROICat' not in d.split('/')[-1]) and
                        (str(subject_id) in d.split('/')[-1]) and
                        ('zstack' not in d.split('/')[-1])])

    session_names = [d.split('/')[-1] for d in raw_paths]
    session_keys = ['_'.join(sn.split('_')[1:3]) for sn in session_names]
    session_inds = np.arange(len(session_names))
    session_types = []
    for raw_path in raw_paths:
        session_json_fn = Path(raw_path) / 'session.json'
        with open(session_json_fn) as f:
            session_json = json.load(f)
        session_types.append(session_json['session_type'])
    session_info_df = pd.DataFrame({'session_name': session_names,
                                'session_key': session_keys,
                                'session_ind': session_inds,
                                'session_type': session_types,
                                'raw_path': raw_paths}) 
    session_info_df.set_index('session_key', drop=True, inplace=True)

    def _map_session_type_to_stimulus(x):
        if any(substring in x for substring in ('gratings', 'STAGE_0', 'STAGE_1')):
            return 'gratings'
        elif 'images_A' in x:
            return 'images_A'
        elif 'images_B' in x:
            return 'images_B'
        else:
            return 'unknown'

    session_info_df['stimulus'] = session_info_df['session_type'].apply(lambda x: _map_session_type_to_stimulus(x))
    image_sets = [s for s in session_info_df.stimulus.unique() if 'images' in s]
    image_order = np.argsort([np.where(session_info_df.stimulus.values == s)[0][0] for s in image_sets])
    # familiar_image = image_sets[image_order[0]]
    # novel_image = image_sets[image_order[1]]
    # count the number of exposure to each stimuli and session_type
    stimulus_exposures = []
    session_type_exposures = []
    for i in range(session_info_df.shape[0]):
        row = session_info_df.iloc[i]
        stimulus_exposures.append(np.where(session_info_df.iloc[:i+1].stimulus.values == row.stimulus)[0].shape[0])
        session_type_exposures.append(np.where(session_info_df.iloc[:i+1].session_type.values == row.session_type)[0].shape[0])
    session_info_df['stimulus_exposures'] = stimulus_exposures
    session_info_df['session_type_exposures'] = session_type_exposures

    return session_info_df


def get_bmod(raw_path):
    ''' Get BehaviorMultiplaneOphysDataset object from a raw path
    ### Currently does not work.
    Use get_bod_list instead.
    '''
    session_name = str(raw_path).split('/')[-1]
    data_dir = Path(raw_path).parent
    processed_dirs = glob.glob(str(data_dir / f'{session_name}*processed*'))
    if len(processed_dirs) == 0:
        raise ValueError(f'No processed data found for session {session_name}')
    elif len(processed_dirs) > 1:
        raise ValueError(f'Multiple processed data found for session {session_name}')
    else:
        session_folder_path = processed_dirs[0]

    eye_dirs = glob.glob(str(data_dir / f'{session_name}*dlc-eye*'))
    if len(eye_dirs) == 0:
        raise ValueError(f'No eye tracking data found for session {session_name}')
    elif len(eye_dirs) > 1:
        raise ValueError(f'Multiple eye tracking data found for session {session_name}')
    else:
        eye_path = eye_dirs[0]
    
    bmod = BehaviorMultiplaneOphysDataset(
        session_folder_path=session_folder_path,
        raw_folder_path=raw_path,
        eye_tracking_path=eye_path,
        pipeline_version='v6'
    )
    return bmod


def get_bod_list(raw_path):
    ''' Get all BehaviorOphysDataset objects from a raw path
    '''
    session_name = str(raw_path).split('/')[-1]
    data_dir = Path(raw_path).parent
    processed_path = list(data_dir.glob(f'{session_name}_processed*'))[0]
    
    opids = []
    for plane_folder in processed_path.glob("*"):
        if plane_folder.is_dir() and not plane_folder.name.startswith("nextflow") \
            and not ('nwb' in plane_folder.name):
            opid = plane_folder.name
            opids.append(opid)

    bod_list = []
    for opid in opids:
        bod = load_plane_data(session_name, opid=opid)
        # cell_specimen_table = get_roi_df_with_valid_roi(bod)
        # bod = add_trials_to_bod(bod)
        session_type = bod.behavior_stimulus_file.session_type
        if 'STAGE_' not in session_type:
            _ = cbu.merge_trials_to_stim_table(bod)
        bod_list.append(bod)
    return bod_list


def get_any_bod(raw_path):
    ''' to get any bod object from a raw path.
    This is mostly for getting session information (e.g., stim_table)
    '''
    session_name = str(raw_path).split('/')[-1]
    data_dir = Path(raw_path).parent
    processed_path = list(data_dir.glob(f'{session_name}_processed*'))[0]

    opids = []
    for plane_folder in processed_path.glob("*"):
        if plane_folder.is_dir() and not plane_folder.name.startswith("nextflow") \
            and not ('nwb' in plane_folder.name):
            opid = plane_folder.name
            break
    bod = load_plane_data(session_name, opid=opid)
    session_type = bod.behavior_stimulus_file.session_type
    if 'STAGE_' not in session_type:
        stim_table = cbu.merge_trials_to_stim_table(bod)
    return bod


def load_plane_data(session_name, opid=None, opid_ind=None, data_dir='/root/capsule/data/',
                    verbose=False):
    ''' Load data using COMB.

    Parameters
    ----------
    session_name : str
        name of the session (e.g., 'multiplane-ophys_721291_2024-05-08_08-05-54')
    opid : str (optional)
        ophys plane ID (e.g., '1365108570', or 'VISp_0')
    opid_ind : int (optional)
        index of the ophys plane ID (e.g., 0)
        if opid is provided, this parameter is ignored
    data_dir : str (optional)
        path to the data directory
    
    Returns
    -------
    bod : BehaviorOphysDataset
        COMB object containing the ophys plane data
    '''

    if opid is None and opid_ind is None:
        raise ValueError('Must provide either opid or opid_ind')
    data_dir = Path(data_dir)
    processed_dirs = glob.glob(str(data_dir / f'{session_name}*processed*'))
    eye_dirs = glob.glob(str(data_dir / f'{session_name}*dlc-eye*'))
    if len(eye_dirs) == 0:
        # raise ValueError(f'No eye tracking data found for session {session_name}')
        print(f'No eye tracking data found for session {session_name}')
        eye_path = None
    elif len(eye_dirs) > 1:
        raise ValueError(f'Multiple eye tracking data found for session {session_name}')
    else:
        eye_path = eye_dirs[0]
        
    if len(processed_dirs) == 0:
        raise ValueError(f'No processed data found for session {session_name}')
    elif len(processed_dirs) > 1:
        raise ValueError(f'Multiple processed data found for session {session_name}')
    else:
        plane_dirs = []
        for path in glob.glob(processed_dirs[0] + '/*'):
            if os.path.isdir(path) and ('nwb' not in path.split('/')[-1]):
                plane_dirs.append(path)

        if opid is not None:
            plane_path = [p for p in plane_dirs if opid in p]
            if len(plane_path) > 1:
                raise ValueError(f'Multiple {opid} found for session {session_name}')
            elif len(plane_path) == 0:
                raise ValueError(f'No {opid} found for session {session_name}')
            else:
                plane_path = plane_path[0]
        else:
            if len(plane_dirs) > opid_ind:
                plane_path = plane_dirs[opid_ind]
                opid = plane_path.split('/')[-1]
                if verbose:
                    print(f'Using plane {opid} for session {session_name}')
            else:
                raise ValueError(f'Processed data for session {session_name} has less than {opid_ind} planes')
    raw_path = Path(plane_path.split('_processed')[0])

    if not raw_path.exists():
        raise ValueError(f'No raw data found for session {session_name}')
    bod = BehaviorOphysDataset(plane_folder_path=plane_path,
                               raw_folder_path=raw_path,
                               eye_tracking_path=eye_path,
                               pipeline_version='v6',
                               verbose=verbose)
    bod.metadata['ophys_plane_id'] = opid
    
    return bod


########################################
## Bypass COMB and get data directly
#########################################

def get_raw_data_dir(session_key, data_dir=Path('/root/capsule/data')):
    data_dir = Path(data_dir)
    raw_path_list = list(data_dir.glob(f'multiplane-ophys_{session_key}_{TIME_FORMAT}'))
    assert len(raw_path_list) == 1, f'Multiple or no raw data found for {session_key}'
    raw_path = raw_path_list[0]
    return raw_path


def get_plane_path_from_session_key_and_plane_id(session_key, plane_id,
                                                data_dir=Path('/root/capsule/data')):
    ''' Get plane path from session key and plane ID
    '''
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    
    processed_list = list(data_dir.glob(f'multiplane-ophys_{session_key}_{TIME_FORMAT}_processed_{DATE_FORMAT}_{TIME_FORMAT}'))
    assert len(processed_list) == 1, f'Multiple processed data found for {session_key}'
    processed_path = processed_list[0]
    plane_path = processed_path / plane_id
    if not os.path.isdir(plane_path):
        raise ValueError(f'No processed data found for {session_key}_{plane_id}')
    return plane_path


def load_dff_from_plane_path(plane_path):
    ''' Load dff for a given plane path
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    '''
    plane_path = Path(plane_path)
    plane_id = plane_path.name
    dff_path = plane_path / 'dff'
    h5_fn = dff_path / f'{plane_id}_dff.h5'
    if not os.path.isfile(h5_fn):
        h5_fn = dff_path / 'dff.h5'
    with h5py.File(h5_fn, 'r') as h:
        dff = h['data'][:]
    return dff


def load_dff(session_key, plane_id,
             data_dir=Path('/root/capsule/data')):
    ''' Load dff for a given session and plane ID
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    '''
    plane_path = get_plane_path_from_session_key_and_plane_id(session_key, plane_id, data_dir=data_dir)
    return load_dff_from_plane_path(plane_path)


def load_raw_roi_fluorescence(session_key, plane_id,
                              data_dir=Path('/root/capsule/data')):
    ''' Load decrosstalked mean image for a given session and plane ID
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    '''
    plane_path = get_plane_path_from_session_key_and_plane_id(session_key, plane_id, data_dir=data_dir)
    extraction_path = plane_path / 'extraction'
    h5_fn = extraction_path / f'{plane_id}_extraction.h5'
    if not os.path.isfile(h5_fn):
        h5_fn = extraction_path / 'extraction.h5'
    with h5py.File(h5_fn, 'r') as h:
        raw_roi_fluourescence = h['traces']['roi'][:]
    return raw_roi_fluourescence


def load_corrected_fluorescence(session_key=None, plane_id=None, plane_path=None,
                                 data_dir = Path('/root/capsule/data')):
    ''' Load corrected fluorescence for a given session and plane ID
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    '''
    if plane_path is None:
        assert session_key is not None and plane_id is not None, \
            'Must provide either plane_path or both session_key and plane_id'
        plane_path = get_plane_path_from_session_key_and_plane_id(session_key, plane_id, data_dir=data_dir)
    else:
        print('Using provided plane_path to load corrected fluorescence')
        plane_id = Path(plane_path).name
    plane_path = Path(plane_path)
    extraction_path = plane_path / 'extraction'
    h5_fn = extraction_path / f'{plane_id}_extraction.h5'
    if not os.path.isfile(h5_fn):
        h5_fn = extraction_path / 'extraction.h5'
    with h5py.File(h5_fn, 'r') as h:
        corrected_fluorescence = h['traces']['corrected'][:]
    return corrected_fluorescence


def load_decrosstalked_mean_image(session_key, plane_id,
                                    data_dir=Path('/root/capsule/data')):
    ''' Load decrosstalked mean image for a given session and plane ID
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    '''
    plane_path = get_plane_path_from_session_key_and_plane_id(session_key, plane_id, data_dir=data_dir)
    extraction_path = plane_path / 'extraction'
    h5_fn = extraction_path / f'{plane_id}_extraction.h5'
    if not os.path.isfile(h5_fn):
        h5_fn = extraction_path / 'extraction.h5'
    with h5py.File(h5_fn, 'r') as h:
        mean_img = h['meanImg'][:]
    return mean_img


def load_projection_image(plane_path, projection_type='mean'):
    """
    Find and load a decrosstalked projection image (mean or max) from the extraction HDF5 file for a given plane path.

    Args:
        plane_path (str or Path): Path to the plane directory containing the extraction HDF5 file.
        projection_type (str, optional): Type of projection image to load. Must be either 'mean' or 'max'. Defaults to 'mean'.

    Returns:
        numpy.ndarray: The requested projection image as a NumPy array.

    Raises:
        AssertionError: If zero or more than one extraction HDF5 file is found in the plane path.
        ValueError: If `projection_type` is not 'mean' or 'max'.
        KeyError: If the requested projection image key is not found in the HDF5 file.
        OSError: If the HDF5 file cannot be opened.

    Example:
        >>> img = load_projection_image('/path/to/plane', projection_type='max')
        >>> print(img.shape)
    """
    # use glob
    plane_path = Path(plane_path)
    extraction_path = list(plane_path.rglob('*_extraction.h5'))
    assert len(extraction_path) == 1, f"Expected exactly 1 extraction file, found {len(extraction_path)} in {plane_path}"
    if projection_type == "mean":
        key = "meanImg"
    elif projection_type == "max":
        key = "maxImg"
    else:
        raise ValueError(f"'projection_type' must be either 'mean' or 'max', instead got {projection_type}")
    with h5py.File(extraction_path[0], 'r') as h:
        img = h[key][:]
    return img


def get_roi_table_from_h5(session_key, plane_id,
                    data_dir = Path('/root/capsule/data')):
    ''' Load ROI table for a given session and plane ID
    It can be retrieved from extraction folder.
    Faster than loading COMB object.

    Filtering NOT applied.
    # TODO: apply filtering?
    '''
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    processed_list = list(data_dir.glob(f'multiplane-ophys_{session_key}*processed*'))
    assert len(processed_list) == 1, f'Multiple processed data found for {session_key}'
    processed_path = processed_list[0]
    plane_path = processed_path / plane_id
    roi_table = get_roi_table_from_plane_path(plane_path)
    return roi_table


def get_roi_table_from_plane_path(plane_path, apply_filter=True, small_roi_radius_threshold_in_um=4):
    ''' Load ROI table for a given plane path
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    '''
    if isinstance(plane_path, str):
        plane_path = Path(plane_path)
    if not os.path.isdir(plane_path):
        raise ValueError(f'Path not found ({plane_path})')
    plane_id = plane_path.name
    extraction_path = plane_path / 'extraction'
    extraction_fn = extraction_path / f'{plane_id}_extraction.h5'
    if not os.path.isfile(extraction_fn):
        extraction_fn = extraction_path / 'extraction.h5'
    if not os.path.isfile(extraction_fn):
        raise ValueError(f'No extraction file found for {plane_id}')
    pixel_masks = file_handling.load_sparse_array(extraction_fn)
            
    roi_table = rois.roi_table_from_mask_arrays(pixel_masks)
    roi_table = roi_table.rename(columns={'id': 'cell_roi_id'})

    if apply_filter:
        range_y, range_x = lamf_utils.get_motion_correction_crop_xy_range(plane_path)

        session_json = get_session_json_from_plane_path(plane_path)
        fov_info = session_json['data_streams'][0]['ophys_fovs'][0] # assume this data is the same for all fovs
        fov_height = fov_info['fov_height']
        fov_width = fov_info['fov_width']
        fov_scale_factor = float(fov_info['fov_scale_factor'])
        
        on_mask = np.zeros((fov_height, fov_width), dtype=bool)
        on_mask[range_y[0]:range_y[1], range_x[0]:range_x[1]] = True
        motion_mask = ~on_mask

        def _touching_motion_border(row, motion_mask):
            if (row.mask_matrix * motion_mask).any():
                return True
            else:
                return False

        roi_table['touching_motion_border'] = roi_table.apply(_touching_motion_border, axis=1, motion_mask=motion_mask)
        
        small_roi_radius_threshold_in_pix = small_roi_radius_threshold_in_um / float(fov_scale_factor)
        area_threshold = np.pi * (small_roi_radius_threshold_in_pix**2)
        
        roi_table['small_roi'] = roi_table['mask_matrix'].apply(lambda x: len(np.where(x)[0]) < area_threshold)
        roi_table['valid_roi'] = ~roi_table['touching_motion_border'] & ~roi_table['small_roi']

    return roi_table


def get_session_json_from_plane_path(plane_path):
    ''' Load session.json for a given plane path
    '''
    if isinstance(plane_path, str):
        plane_path = Path(plane_path)
    if not os.path.isdir(plane_path):
        raise ValueError(f'Path not found ({plane_path})')
    try:
        session_json_fn = next(plane_path.parent.rglob('*session.json'))
    except StopIteration:
        session_name = plane_path.parent.name.split('_processed')[0]
        raw_path = plane_path.parent.parent / session_name
        session_json_fn = raw_path / 'session.json'
    with open(session_json_fn) as f:
        session_json = json.load(f)
    return session_json


def get_suite2p_ops_from_plane_path(plane_path):
    ''' Load suite2p ops for a given plane path
    '''
    if isinstance(plane_path, str):
        plane_path = Path(plane_path)
    if not os.path.isdir(plane_path):
        raise ValueError(f'Path not found ({plane_path})')
    motion_correction_json_path = list((plane_path / 'motion_correction').glob('*_motion_correction_data_process.json'))
    if len(motion_correction_json_path) == 0:
        motion_correction_json_path = list((plane_path / 'motion_correction').glob('processing.json'))
        if len(motion_correction_json_path) == 0:
            raise ValueError(f'No extraction json found for {plane_path}')
    elif len(motion_correction_json_path) > 1:
        raise ValueError(f'Multiple extraction json found for {plane_path}')
    with open(motion_correction_json_path[0]) as f:
        motion_correction_json = json.load(f)
    if 'parameters' in motion_correction_json:
        suite2p_ops = motion_correction_json['parameters']['suite2p_args']
    elif 'processing_pipeline' in motion_correction_json:
        suite2p_ops = motion_correction_json['processing_pipeline']['data_processes'][0]['parameters']['suite2p_args']
    return suite2p_ops


def get_frame_rate_from_plane_path(plane_path):
    ''' Load frame rate for a given plane path
    '''
    if isinstance(plane_path, str):
        plane_path = Path(plane_path)
    if not os.path.isdir(plane_path):
        raise ValueError(f'Path not found ({plane_path})')
    session_json = get_session_json_from_plane_path(plane_path)
    frame_rate = float(session_json['data_streams'][0]['ophys_fovs'][0]['frame_rate'])
    return frame_rate


def get_decrosstalked_movie_file(plane_path):
    ''' Load decrosstalked movie for a given plane path
    It can be retrieved from extraction folder.
    Faster than loading COMB object.
    '''
    if isinstance(plane_path, str):
        plane_path = Path(plane_path)
    if not os.path.isdir(plane_path):
        raise ValueError(f'Path not found ({plane_path})')
    plane_id = plane_path.name
    decrosstalk_path = plane_path / 'decrosstalk'
    decrosstalked_movie_fn = decrosstalk_path / f'{plane_id}_decrosstalk.h5'
    if not os.path.isfile(decrosstalked_movie_fn):
        raise ValueError(f'No decrosstalked movie found for {plane_id}')    
    return decrosstalked_movie_fn

