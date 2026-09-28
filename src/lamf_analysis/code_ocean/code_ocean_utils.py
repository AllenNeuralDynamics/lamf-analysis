import os
import numpy as np
import pandas as pd
import warnings
import json
import glob
from pathlib import Path
import h5py
import time
from datetime import datetime, timezone
import requests
from typing import Union
import re

from codeocean import CodeOcean
from codeocean.data_asset import (DataAssetSearchParams,
                                  DataAssetAttachParams)
from codeocean.components import SearchFilter

from lamf_analysis.code_ocean import capsule_bod_utils as cbu
import lamf_analysis.utils as lamf_utils
from lamf_analysis.code_ocean import docdb_utils
from lamf_analysis.code_ocean import s3_utils

import logging
logger = logging.getLogger(__name__)

def get_co_client():
    domain="https://codeocean.allenneuraldynamics.org/"
    token = os.getenv('API_SECRET')
    if token is None:
        token = os.getenv('CUSTOM_KEY')
    client = CodeOcean(domain=domain, token=token)
    t_or_f, msg = check_co_client_token(client)
    if not t_or_f:
        logger.warning(f"CodeOcean client token check failed: {msg}")
    return client


def check_co_client_token(client):
    # Check if the provided CodeOcean client has a valid token by making a test API call.
    try:
        params = DataAssetSearchParams(offset=0, limit=1, sort_order='desc',
                                       sort_field='name', type='dataset',
                                       archived=False, favorite=False, query='')
        client.data_assets.search_data_assets(params)
        return True, "API call succeeded (token accepted)"
    except Exception as exc:
        msg = str(exc)
        if '401' in msg or 'Unauthorized' in msg or '403' in msg:
            return False, f"authentication failure: {msg}"
        return False, f"API call failed: {msg}"


def get_data_asset_id_from_name(asset_name,
                                data_type: str = Union['raw', 'derived'],
                                client=None):
    ''' Get data asset ID from CodeOcean by name
    example:
        asset_name = '1299958728_2022-03-15_13-28-02_multiplane-ophys_raw'
        asset_id = get_data_asset_id_from_name(asset_name)
    '''
    if client is None:
        client = get_co_client()
    assert data_type in ['raw', 'derived'], f"Invalid data_type '{data_type}', expected 'raw' or 'derived'"
    search_type = 'dataset' if data_type == 'raw' else 'result'
    data_asset_params = DataAssetSearchParams(
        offset=0,
        limit=None,
        sort_order="desc",
        sort_field="created",
        type=search_type,
        archived=False,
        favorite=False,
        query=asset_name,
    )
    results = client.data_assets.search_data_assets(data_asset_params).results
    if len(results) == 0:
        raise ValueError(f"No {data_type} asset found matching name '{asset_name}'")
    elif len(results) > 1:
        print(f"Warning: multiple {data_type} assets found matching name '{asset_name}', returning newest match")
    return results[0].id


def get_co_raw_id_from_name(raw_name, client=None, multiple_allowed=False):
    ''' Get raw data asset ID from CodeOcean by name
    example:
        raw_name = '1299958728_2022-03-15_13-28-02_multiplane-ophys_raw'
        raw_id = get_co_raw_id_from_name(raw_name)
    '''
    if client is None:
        client = get_co_client()
    data_asset_params = DataAssetSearchParams(
        offset=0,
        limit=None,
        sort_order="desc",
        sort_field="name",
        type="dataset",
        archived=False,
        favorite=False,
        query=raw_name,
    )
    results = client.data_assets.search_data_assets(data_asset_params).results
    if not multiple_allowed:
        assert len(results) == 1, (
            f"Expected exactly one raw asset matching name '{raw_name}', "
            f"found {len(results)}"
        )
    assert 'raw' in results[0].tags
    raw_id = results[0].id
    return raw_id


def get_data_asset_search_results(query_str, subject_id=None):
    ''' Get data asset search results from CodeOcean
    example: 
        query_str = f'conditioned_mean_response_v2' # works for data asset name beginning with this string
        results = get_data_asset_search_results(query_str, subject_id)
    '''
    client = get_co_client()
    data_asset_params = DataAssetSearchParams(
        offset=0,
        limit=None,
        sort_order="desc",
        sort_field="name",
        type="dataset",
        archived=False,
        favorite=False,
        query=query_str
    )
    data_assets = client.data_assets.search_data_assets(data_asset_params)
    if subject_id is not None:
        results = [da for da in data_assets.results if f'{subject_id}' in da.name]
    else:
        results = data_assets.results
    return results


def set_data_asset_params(subject_id, data_name='multiplane-ophys', tags=['raw'],
                          offset=0, limit=1000):
    data_asset_filters = [        
        SearchFilter(
            key="tags",
            value=str(subject_id)
        ),
    ]
    if data_name != '':
        data_asset_filters.append(
            SearchFilter(
                key="name",
                value=data_name
            )
        )
    data_asset_filters.extend([SearchFilter(
            key="tags",
            value=tag) for tag in tags
        ])
    
    data_asset_params = DataAssetSearchParams(
        offset=offset,
        limit=limit,
        sort_order="desc",
        sort_field="name",
        archived=False,
        favorite=False,
        # query="name:'multiplane-ophys'",
        filters=data_asset_filters
    )
    return data_asset_params


def get_mouse_sessions_by_filters(subject_id, data_name='multiplane-ophys',
                                  offset=0, limit=1000,
                                  filter_test_data=False,
                                  docdb_api_client=None):
    """Return raw-session metadata for one subject from DocDB.

    This is the DocDB-backed replacement for the former function of the same
    name, which returned ``aind_session.Session`` objects.  It returns a
    dataframe instead, with one raw session per row and session metadata,
    Code Ocean raw-asset IDs, and DocDB-recorded S3 locations.

    ``offset`` and ``limit`` retain their former pagination meaning after
    DocDB filtering.  Set ``filter_test_data=True`` to apply the standard
    test-session exclusion used by ``get_session_infos_from_docdb``.
    """
    session_df = docdb_utils.get_session_infos_from_docdb(
        subject_id=subject_id,
        docdb_api_client=docdb_api_client,
        data_type=data_name,
        filter_test_data=filter_test_data,
        filter_by_genotype=False,
        docdb_version="v1",
    )
    if session_df is None:
        return pd.DataFrame()

    end = None if limit is None else offset + limit
    return session_df.iloc[offset:end].reset_index(drop=True)


def get_derived_assets_df(subject_id, process_name,
                       data_name='multiplane-ophys',
                       offset=0, limit=1000,
                       processing_parameters=None,
                       add_s3_location=True):    
    results = get_derived_assets(subject_id, process_name,
                                data_name=data_name,
                                offset=offset, limit=limit,
                                processing_parameters=processing_parameters)
    
    process_name_lower = process_name.lower()
    derived_asset_rows = []
    for res in results:
        name = res.name
        name_lower = name.lower()
        raw_asset_name = name_lower.split(process_name_lower)[0].rstrip('_')
        if data_name != '':
            session_name = '_'.join(raw_asset_name.split('_')[1:3])
        else:
            session_name = '_'.join(raw_asset_name.split('_')[0:2])
        if str(subject_id) != session_name.split('_')[0]:
            raise ValueError(f"Subject ID mismatch in asset '{name}': expected {subject_id}, found {session_name.split('_')[0]}")
        ts = res.created
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        s = dt_utc.strftime("%Y-%m-%d_%H-%M-%S")
        derived_asset_rows.append({
            'derived_asset_id': res.id,
            'derived_asset_name': name,
            'raw_asset_name': raw_asset_name,
            'session_name': session_name,
            'derived_date': s.split('_')[0],
            'derived_time': s.split('_')[1],
        })
    derived_asset_df = pd.DataFrame(derived_asset_rows,
                                    columns=['derived_asset_id',
                                            'derived_asset_name',
                                            'raw_asset_name',
                                            'session_name',
                                            'derived_date',
                                            'derived_time'])
    if add_s3_location:
        s3_paths = docdb_utils.get_s3_paths_for_code_ocean_assets(
            derived_asset_df['derived_asset_id'].tolist())
        derived_asset_df['s3_path'] = derived_asset_df['derived_asset_id'].map(s3_paths)
    return derived_asset_df


def get_derived_assets(subject_id, process_name,
                       data_name='multiplane-ophys',
                       offset=0, limit=1000,
                       subprocessing_name=None,
                       processing_parameters=None):
    client = get_co_client()
    tags = ['derived', process_name]
    results = []
    search_offset = offset
    while True:
        data_asset_params = set_data_asset_params(subject_id=subject_id, 
                                                  data_name=data_name, tags=tags,
                                                  offset=search_offset, limit=limit)
        data_asset_search_results = client.data_assets.search_data_assets(data_asset_params)
        if processing_parameters is not None:
            assert subprocessing_name is not None, "Must specify subprocessing_name when filtering by processing_parameters"
            for r in data_asset_search_results.results:
                data_asset_id = r.id
                parameters = get_process_parameters(data_asset_id, subprocessing_name)
                if parameters is not None:
                    check_parameter_list = [val == parameters.get(key) for key, val in processing_parameters.items()]
                    if np.all(check_parameter_list):
                        results.append(r)
        else:
            results.extend(data_asset_search_results.results)
        if not data_asset_search_results.has_more:
            break
        search_offset += limit
    return results


def get_process_parameters(data_asset_id, process_name, processing_json_path='processing.json'):
    client = get_co_client()
    try:
        url = client.data_assets.get_data_asset_file_urls(data_asset_id, path=processing_json_path).download_url
        assert url is not None
        processes = requests.get(url).json()['processing_pipeline']['data_processes']
        process_names = [p.get("name", None) for p in processes]        
        if process_name in process_names:
            process_ind = process_names.index(process_name)        
            parameters = processes[process_ind].get('parameters', None)
        else:
            print(f"Process name '{process_name}' not found in processing pipeline for data asset '{data_asset_id}'")
            parameters = None
        return parameters
    except:
        print('Cannot get processing parameters')
        return None


def check_process_names(data_asset_id, processing_json_path='processing.json'):
    client = get_co_client()
    try:
        url = client.data_assets.get_data_asset_file_urls(data_asset_id, path=processing_json_path).download_url
        assert url is not None
        process_names = [p.get("name", None) for p in requests.get(url).json()['processing_pipeline']['data_processes']]
        return process_names
    except:
        print('Cannot get processing parameters')
        return None


# def get_processing_parameters(data_asset_id, processing_json_path='processing.json'):
#     client = get_co_client()
#     try:
#         url = client.data_assets.get_data_asset_file_urls(data_asset_id, path=processing_json_path).download_url
#         assert url is not None
#         parameters = requests.get(url).json()['processing_pipeline']['data_processes'][0]['parameters']
#         return parameters
#     except:
#         print('Cannot get processing parameters')
#         return None


def get_input_asset_provenance(asset_id: str):
    client = get_co_client()
    asset = client.data_assets.get_data_asset(asset_id)

    input_ids = (asset.provenance.data_assets
                 if asset.provenance and asset.provenance.data_assets
                 else [])

    inputs = []
    for input_id in input_ids:
        da = client.data_assets.get_data_asset(input_id)
        inputs.append({"id": da.id, "name": da.name, "type": str(da.type)})

    return pd.DataFrame(inputs), asset.provenance.computation if asset.provenance else None



def get_hcr_processed_data_assets(subject_id,
                                  co_client=None):
    """
    Retrieve HCR processed data assets for a given subject from CodeOcean.

    Parameters
    ----------
    subject_id : str
        The identifier of the subject for which to retrieve HCR data assets.
    co_client : CodeOcean, optional
        An instance of the CodeOcean client. If None, a new client will be created.

    Returns
    -------
    search_results : DataAssetSearchResults
        The search results object containing the matching HCR processed data assets.
    """
    if co_client is None:
        co_client = get_co_client()

    data_asset_filters = [
        SearchFilter(
            key="tags",
            value="HCR"
        ),
        SearchFilter(
            key="tags",
            value="processed"
        ),
        SearchFilter(
            key="name",
            value=f"HCR_{subject_id}"
        ),
    ]
    data_asset_params = DataAssetSearchParams(
            offset=0,
            limit=1000,
            sort_order="desc",
            sort_field="name",
            archived=False,
            favorite=False,
            # query="name:'multiplane-ophys'",
            filters=data_asset_filters
        )

    search_results = co_client.data_assets.search_data_assets(data_asset_params)
    return search_results
  

def attach_assets(assets: list, co_client=None):
    """Attach list of asset_ids to capsule with CodeOcean SDK, print mount state
    
    Parameters
    ----------
    assets : list
        list of asset_ids
        Example: ['1az0c240-1a9z-192b-pa4c-22bac5ffa17b', '1az0c240-1a9z-192b-pa4c-22bac5ffa17b']
    co_client : object
        CodeOcean client object
        If None, must set "API_SECRET" in environment variable for CodeOcean token
        
    Returns
    -------
    None
    """
    
    if co_client is None:
        co_client = get_co_client()

    # DataAssetAttachParams(id="1az0c240-1a9z-192b-pa4c-22bac5ffa17b", mount="Reference")
    data_assets = [DataAssetAttachParams(id=aid) for aid in assets]        
            
    results = co_client.computations.attach_data_assets(
        computation_id=os.getenv("CO_COMPUTATION_ID"),
        attach_params=data_assets,
    )

    for target_id in assets:
        result = next((item for item in results if item.id == target_id), None)

        if result:
            ms = result.mount_state
            logger.info(f"asset_id: {target_id} - mount_state: {ms}")
            print(f"asset_id: {target_id} - mount_state: {ms}")
        else:
            print(f"asset_id: {target_id} - not found in CodeOcean API response")
    return


def archive_data_assets(asset_ids, archive=True, dry_run=False, co_client=None):
    """Archive or unarchive one or more Code Ocean data assets.

    Parameters
    ----------
    asset_ids : str | list[str]
        A single data asset id or list of data asset ids.
    archive : bool, optional
        True to archive assets, False to unarchive.
    dry_run : bool, optional
        If True, do not call API; only print what would be changed.
    co_client : CodeOcean, optional
        An instance of the CodeOcean client. If None, a new client is created.

    Returns
    -------
    list[dict]
        Per-asset operation results with keys: asset_id, action, status, error.
    """
    if co_client is None:
        co_client = get_co_client()

    if isinstance(asset_ids, str):
        asset_ids = [asset_ids]
    elif asset_ids is None:
        asset_ids = []
    else:
        asset_ids = list(asset_ids)

    unique_asset_ids = list(dict.fromkeys(asset_ids))
    action = "archive" if archive else "unarchive"
    results = []

    for asset_id in unique_asset_ids:
        if dry_run:
            logger.info(f"[DRY RUN] would {action} asset_id={asset_id}")
            print(f"[DRY RUN] would {action} asset_id={asset_id}")
            results.append({
                "asset_id": asset_id,
                "action": action,
                "status": "dry_run",
                "error": None,
            })
            continue

        try:
            co_client.data_assets.archive_data_asset(data_asset_id=asset_id, archive=archive)
            logger.info(f"{action}d asset_id={asset_id}")
            print(f"{action}d asset_id={asset_id}")
            results.append({
                "asset_id": asset_id,
                "action": action,
                "status": "ok",
                "error": None,
            })
        except Exception as exc:
            logger.error(f"failed to {action} asset_id={asset_id}: {exc}")
            print(f"failed to {action} asset_id={asset_id}: {exc}")
            results.append({
                "asset_id": asset_id,
                "action": action,
                "status": "error",
                "error": str(exc),
            })

    return results
