import os
import numpy as np
import pandas as pd
import warnings
import json
import glob
from pathlib import Path
import h5py
import time

from codeocean import CodeOcean
from codeocean.data_asset import (DataAssetSearchParams,
                                  DataAssetAttachParams)
from codeocean.components import SearchFilter

import aind_session
from aind_session import Session

from lamf_analysis.code_ocean import capsule_bod_utils as cbu
import lamf_analysis.utils as lamf_utils
from lamf_analysis.code_ocean import docdb_utils

import logging
logger = logging.getLogger(__name__)

DEFAULT_MOUNT_TO_IGNORE = ['fb4b5cef-4505-4145-b8bd-e41d6863d7a9', # Ophys_Extension_schema_10_14_2024_13_44
                            '35d1284e-4dfa-4ac3-9ba8-5ea1ae2fdaeb'], # ROI classifier V1
TIME_FORMAT = '[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'
DATE_FORMAT = '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'


def get_co_client():
    domain="https://codeocean.allenneuraldynamics.org/"
    token = os.getenv('API_SECRET')
    client = CodeOcean(domain=domain, token=token)
    return client


def get_co_raw_id_from_name(raw_name, client=None):
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
                                  offset=0, limit=1000):
    client = get_co_client()

    results = []
    while True:
        data_asset_params = set_data_asset_params(subject_id=subject_id, 
                                                  data_name=data_name, tags=['raw'],
                                                  offset=offset, limit=limit)
        data_asset_search_results = client.data_assets.search_data_assets(data_asset_params)
        results.extend(data_asset_search_results.results)
        if not data_asset_search_results.has_more:
            break
        data_asset_params.offset += data_asset_params.limit
    
    sessions = set()
    for result in results:
        name = result.name
        session = Session(name)
        sessions.add(session)
    sessions = tuple(sorted(sessions, key=lambda s: s.dt))
    return sessions


def get_derived_assets_df(subject_id, process_name,
                       data_name='multiplane-ophys',
                       offset=0, limit=1000,
                       add_s3_location=True):
    results = get_derived_assets(subject_id, process_name,
                                data_name=data_name,
                                offset=offset, limit=limit)
    
    derived_asset_rows = []
    for res in results:
        name = res.name
        raw_asset_name = name.split(process_name)[0].rstrip('_')
        if data_name != '':
            session_name = '_'.join(raw_asset_name.split('_')[1:3])
        else:
            session_name = '_'.join(raw_asset_name.split('_')[0:2])
        if str(subject_id) != session_name.split('_')[0]:
            raise ValueError(f"Subject ID mismatch in asset '{name}': expected {subject_id}, found {session_name.split('_')[0]}")
        derived_asset_rows.append({
            'derived_asset_id': res.id,
            'derived_asset_name': name,
            'raw_asset_name': raw_asset_name,
            'session_name': session_name,
        })
    derived_asset_df = pd.DataFrame(derived_asset_rows,
                                    columns=['derived_asset_id',
                                            'derived_asset_name',
                                            'raw_asset_name',
                                            'session_name'])
    if add_s3_location:
        derived_asset_df['s3_path'] = derived_asset_df['derived_asset_id'].apply(
            aind_session.utils.get_source_dir_by_name)
    return derived_asset_df


def get_derived_assets(subject_id, process_name,
                       data_name='multiplane-ophys',
                       offset=0, limit=1000):
    client = get_co_client()
    tags = ['derived', process_name]
    results = []
    while True:
        data_asset_params = set_data_asset_params(subject_id=subject_id, 
                                                  data_name=data_name, tags=tags,
                                                  offset=offset, limit=limit)
        data_asset_search_results = client.data_assets.search_data_assets(data_asset_params)
        results.extend(data_asset_search_results.results)
        if not data_asset_search_results.has_more:
            break
        data_asset_params.offset += data_asset_params.limit
    return results


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
