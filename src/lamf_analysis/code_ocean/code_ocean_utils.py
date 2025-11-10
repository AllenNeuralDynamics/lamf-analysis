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
from aind_ophys_data_access import capsule
from comb.behavior_ophys_dataset import BehaviorOphysDataset, BehaviorMultiplaneOphysDataset
from aind_ophys_data_access import rois
from comb import file_handling

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


def get_data_asset_search_results(query_str, mouse_id=None):
    ''' Get data asset search results from CodeOcean
    example: 
        query_str = f'conditioned_mean_response_v2' # works for data asset name beginning with this string
        results = get_data_asset_search_results(query_str, mouse_id)
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
    if mouse_id is not None:
        results = [da for da in data_assets.results if f'{mouse_id}' in da.name]
    else:
        results = data_assets.results
    return results


def set_data_asset_params(mouse_id, data_name='multiplane-ophys', tags=['raw'],
                          offset=0, limit=1000):
    data_asset_filters = [        
        SearchFilter(
            key="tags",
            value=str(mouse_id)
        ),
        SearchFilter(
            key="name",
            value=data_name
        ),
    ]
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


def get_mouse_sessions_by_filters(mouse_id, data_name='multiplane-ophys', data_level='raw',
                                  offset=0, limit=1000):
    client = get_co_client()

    results = []
    while True:
        data_asset_params = set_data_asset_params(mouse_id=mouse_id, 
                                                  data_name=data_name, tags=[data_level],
                                                  offset=offset, limit=limit)
        data_asset_search_results = client.data_assets.search_data_assets(data_asset_params)
        results.extend(data_asset_search_results.results)
        if ~data_asset_search_results.has_more:
            break
        data_asset_params.offset += data_asset_params.limit
    
    sessions = set()
    for restuls in results:
        name = restuls.name
        session = Session(name)
        sessions.add(session)
    sessions = tuple(sorted(sessions, key=lambda s: s.dt))
    return sessions


def get_derived_assets(mouse_id, process_name,
                       data_name='multiplane-ophys',
                       offset=0, limit=1000):
    client = get_co_client()
    tags = ['derived', process_name]
    data_asset_params = set_data_asset_params(mouse_id=mouse_id, data_name=data_name, tags=tags,
                                              offset=offset, limit=limit)
    results = []
    while True:
        data_asset_params = set_data_asset_params(mouse_id=mouse_id, 
                                                  data_name=data_name, tags=tags,
                                                  offset=offset, limit=limit)
        data_asset_search_results = client.data_assets.search_data_assets(data_asset_params)
        results.extend(data_asset_search_results.results)
        if ~data_asset_search_results.has_more:
            break
        data_asset_params.offset += data_asset_params.limit
    return results