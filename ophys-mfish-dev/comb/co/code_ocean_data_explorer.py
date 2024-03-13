"""Class to find data assets in code ocean for visual behavior like ophys and behavior data."""
import os
import pandas as pd
import numpy as np
from typing import Optional


from aind_codeocean_api.codeocean import CodeOceanClient, CodeOceanCredentials


class CodeOceanDataExplorer(object):
    """Class to find data assets in code ocean for visual behavior like ophys and behavior data."""
    def __init__(self,
                 ophys_plane_id: Optional[str] = None,
                 behavior_session_id: Optional[str] = None):

        self.ophys_plane_id = ophys_plane_id
        self.behavior_session_id = behavior_session_id

        self.client = self._get_client()

        self.all_data_assets = self._get_all_multiplane_data_assets()

    
    def _get_client(self):
        token, domain = self._code_ocean_credentials()
        client = CodeOceanClient(domain=domain, token=token)
        return client

    def _get_env_var(self,var_name):
        try:
            return os.environ[var_name]
        except KeyError:
            raise KeyError(f"Environment variable {var_name} not found")

    def _code_ocean_credentials(self):
        token = self._get_env_var("CODEOCEAN_TOKEN")
        domain = self._get_env_var("CODEOCEAN_DOMAIN")
        return token, domain

    def _get_all_multiplane_data_assets(self):

        query = "multiplane"
        client = self.client

        response = client.search_all_data_assets(query=query)
        mp = response.json()
        results = mp['results']

        return results

    def _filtered_by_tags(self, results, tag, return_no_tag = False):
        filtered_assets=[]
        no_tags_assets = []
        for r in results:
            try:
                if tag in r['tags']:
                    filtered_assets.append(r)
            except KeyError:
                no_tags_assets.append(r)
                continue
        if return_no_tag:
            return filtered_assets, no_tags_assets
        else:
            return filtered_assets

    def get_derived_multiplane_data_assets(self):
        derived = self._filtered_by_tags(self.all_data_assets, 'derived')
        return derived

    def get_raw_multiplane_data_assets(self):
        raw = self._filtered_by_tags(self.all_data_assets, 'raw')
        return raw

    def _get_data_assets_by_type(self, asset_type):
        assert asset_type in ["all", "raw", "derived"], "asset_type must be one of 'all', 'raw', or 'derived'"
        if asset_type == "all":
            data_assets_list = self._get_all_multiplane_data_assets()
        elif asset_type == "raw":
            data_assets_list = self._get_raw_multiplane_data_assets()
        elif asset_type == "derived":
            data_assets_list = self._get_derived_multiplane_data_assets()

        return data_assets_list

    def mouse_id_data_assets(self, mouse_id, asset_type="all"):
        """addd

        Parameters
        ----------
        mouse_id : str
            Mouse id to filter data assets.
        asset_type : str, optional
            Type of asset to filter. The default is "all".  Options are "all", "raw", "derived".

        Returns
        -------
        list
            List of data assets filtered by mouse id.
        """
        data_assets_list = self._get_data_assets_by_type(asset_type)

        mouse_id_assets = self._filtered_by_tags(data_assets_list, mouse_id)
        return mouse_id_assets

