# %%
from pathlib import Path
import json
import pandas as pd

import lamf_analysis.code_ocean_scripts.jobs as jobs
import aind_ophys_data_access.session_utils as session_utils

import codeocean


# %%



# %%
# %% CODE O
from codeocean.components import SearchFilter, SearchFilterRange
from codeocean.data_asset import DataAssetSearchParams
import os
import datetime
client = codeocean.CodeOcean(domain=os.getenv("CODEOCEAN_DOMAIN"), 
                             token=os.getenv("CODEOCEAN_TOKEN"))


def search_data_assets_page(query_params, offset=0, limit=1000):
    """
    Execute a single data asset search request with the given parameters.
    
    Parameters:
    -----------
    query_params : dict
        Dictionary of search parameters
    offset : int
        Starting position for pagination
    limit : int
        Maximum number of results to return (API may cap at 1000)
        
    Returns:
    --------
    list
        Search results for the current page
    """
    # Create search parameters with pagination values
    data_asset_params = DataAssetSearchParams(
        offset=offset,
        limit=limit,
        **query_params
    )
    
    # Execute the search
    response = client.data_assets.search_data_assets(data_asset_params)
    return response.results


def query_assets_with_pagination(query_params, page_size=1000, verbose=True):
    """
    Retrieve all data assets matching the given parameters by handling pagination.
    
    Parameters:
    -----------
    query_params : dict
        Dictionary of search parameters
    page_size : int
        Size of each page/batch to retrieve (max 1000)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    list
        All search results combined
    """
    offset = 0
    all_results = []
    more_results = True
    
    while more_results:
        # Get current page of results
        current_page_results = search_data_assets_page(query_params, offset, page_size)
        
        # Add results to our collection
        all_results.extend(current_page_results)
        
        # Check if we should continue or if we've reached the end
        if len(current_page_results) < page_size:
            more_results = False
        else:
            offset += page_size
        
        if verbose:
            print(f"Retrieved {len(current_page_results)} results at offset {offset-page_size}")
    
    if verbose:
        print(f"Total results: {len(all_results)}")
    
    return all_results


def multiplane_ophys_assets(query_prefix="multiplane-ophys",
                            type=None,
                            sort_order="desc", 
                            sort_field="name", 
                            verbose=True):
    """
    Retrieve all multiplane ophys assets for a specific subject ID.
    
    Parameters:
    -----------
    subject_id : str
        The subject ID to filter assets by (e.g., "HCR")
    query_prefix : str
        The query prefix to search for (e.g., "multiplane-ophys_")
    sort_order : str
        The sort order for results (e.g., "asc" or "desc")
    sort_field : str
        The field to sort results by (e.g., "name", "created_at")
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    list
        All matching multiplane ophys assets
    """
    # Define search parameters
    query_params = {
        "sort_order": sort_order,
        "sort_field": sort_field,
        "archived": False,
        "favorite": False,
        "query": query_prefix,
        "type": type,
        #"filters": [SearchFilter(key="subject id", value=subject_id)]
    }
    
    # Get all matching data assets using pagination
    return query_assets_with_pagination(query_params,verbose=verbose)

def subject_ids_from_assets(assets):
    """
    Get all subject IDs from a list of assets
    """
    subject_ids = []
    for asset in assets:
        try:
            subject_id = asset.custom_metadata.get("subject id")
            if subject_id:
                subject_ids.append(subject_id)
        except AttributeError:
            continue
    subject_ids = list(set(subject_ids))

    # drop any bigger than 6 chars
    subject_ids = [subject_id for subject_id in subject_ids if len(subject_id) <= 6]
    subject_ids
    print(f"Found {len(subject_ids)} subject IDs")
    return subject_ids


# raw are dataset type here
assets = multiplane_ophys_assets(type="dataset")
multiplane_subject_ids = subject_ids_from_assets(assets)
print(multiplane_subject_ids)

# %%


# %%
assets = session_utils.get_asset_type_from_sessions_list(sessions, "raw")
assets_dict = [[{"id": asset.id, "mount": asset.mount}] for asset in assets]
assets_dict