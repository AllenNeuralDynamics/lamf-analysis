"""Class to find data assets in code ocean for visual behavior like ophys and behavior data."""
import os
import pandas as pd
import numpy as np
from typing import Optional


from aind_codeocean_api.codeocean import CodeOceanClient, CodeOceanCredentials


class CodeOceanDataExplorer(object):
    """Class to find data assets in code ocean for visual behavior like ophys and behavior data."""
    def __init__(self,
                 query: Optional[str] = None,
                 verbose: Optional[bool] = True):

        self.client = self._get_client()
        self.query = query
        self.all_data_assets = self._all_data_assets()
        self.verbose = verbose

        if self.verbose:
            print("CodeOceanDataExplorer initialized\n---------------------------------")
            print(f"Query: {query}")
            print(f"Number of assets: {len(self.all_data_assets)}")
        

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

    def _all_data_assets(self):
        client = self.client

        response = client.search_all_data_assets(query=self.query)
        mp = response.json()
        results = mp['results']

        return results

    @property
    def derived_assets(self):
        derived = self.filter_by_tag(self.all_data_assets, 'derived')
        return derived

    @property
    def raw_assets(self):
        raw = self.filter_by_tag(self.all_data_assets, 'raw')
        return raw

    def filter_by_tag(self, results, tag, return_no_tag = False):
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
    

    def _get_data_assets_by_type(self, asset_type):
        assert asset_type in ["all", "raw", "derived"], "asset_type must be one of 'all', 'raw', or 'derived'"
        if asset_type == "all":
            data_assets_list = self._all_data_assets()
        elif asset_type == "raw":
            data_assets_list = self.raw_assets
        elif asset_type == "derived":
            data_assets_list = self.derived_assets

        return data_assets_list

    def assets_by_mouse_id(self, mouse_id, asset_type="all"):
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

        mouse_id_assets = self.filter_by_tag(data_assets_list, mouse_id)
        return mouse_id_assets

    def assets_by_name(self, name, asset_type="all"):
        """addd

        Parameters
        ----------
        name : str
            Name of data asset to filter.
        asset_type : str, optional
            Type of asset to filter. The default is "all".  Options are "all", "raw", "derived".

        Returns
        -------
        list
            List of data assets filtered by name.
        """
        data_assets_list = self._get_data_assets_by_type(asset_type)

        # iterate over all data assets and find the ones that contain the string name in "name"
        name_assets = []
        for r in data_assets_list:
            if name in r['name']:
                name_assets.append(r)

        return name_assets


    def assets_by_base_name(self, basename, asset_type="all"):
        """Grab assests by base_name (i.e. session name)

        In AIND data assets are named with the following convention:
        raw data: {platform}_{mouse_id}_{date}_{time}
        derived data: {platform}_{mouse_id}_{date}_{time}_{derived_data_type}_{date}_{time}
        Therefore, base_name the first 4 elements of the name split by "_"

        Parameters
        ----------
        basename : str
            Basename of data asset to filter.
        asset_type : str, optional
            Type of asset to filter. The default is "all".  Options are "all", "raw", "derived".

        Returns
        -------
        list
            List of data assets filtered by basename.
        """
        data_assets_list = self._get_data_assets_by_type(asset_type)

        basename_assets = []
        for r in data_assets_list:
            name = "_".join(r['name'].split("_")[:4])
            if basename == name:
                basename_assets.append(r)
        return basename_assets


    def assets_by_session_name(self, session_name, asset_type="all"):
        """Grab assests by session_name

        See: assets_by_base_name()

        Parameters
        ----------
        session_name : str
            Session name of data asset to filter.
        asset_type : str, optional
            Type of asset to filter. The default is "all".  Options are "all", "raw", "derived".

        Returns
        -------
        list
            List of data assets filtered by session name.
        """
        return self.assets_by_base_name(session_name, asset_type)


    def assets_by_platform(self, platform, asset_type="all"):
        """Grab assests by platform

        Parameters
        ----------
        platform : str
            Platform of data asset to filter.
        asset_type : str, optional
            Type of asset to filter. The default is "all".  Options are "all", "raw", "derived".

        Returns
        -------
        list
            List of data assets filtered by platform.
        """
        data_assets_list = self._get_data_assets_by_type(asset_type)

        platform_assets = []
        for r in data_assets_list:
            name = r['name'].split("_")[0]
            if platform == name:
                platform_assets.append(r)
        return platform_assets

