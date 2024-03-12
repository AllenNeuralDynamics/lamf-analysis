from .code_ocean_data_explorer import CodeOceanDataExplorer

import os
import yaml
import requests
from typing import Optional, Union
from pathlib import Path








class DataAssetLoader(CodeOceanDataExplorer):
    """

    For multi-plane data assets only

    E.g.
    raw: 'multiplane-ophys_677594_2023-08-04_09-44-08'
    Processed: 'multiplane-ophys_677594_2023-08-04_09-44-08_processed_2024-02-08_23-26-44'

    Parameters
    ----------
    """
    def __init__(self):
        super().__init__()

        # attached_dict
        self.attached_dict = self.get_attached_data_assets()

        # adds "linked_attached" key; bool that indicates in linked (raw/processed) data asset is attached
        self._update_linked_key()



    def linked_data_assets(self):
        return {k:v for k,v in self.attached_dict.items() if v["linked_attached"]}

    def unlinked_data_assets(self):
        return {k:v for k,v in self.attached_dict.items() if not v["linked_attached"]}

    def _update_linked_key(self):
        
        for asset in self.attached_dict:
            check_dict = self.attached_dict.copy()
            check_dict.pop(asset)
            # remove asset from check dict

            self.attached_dict[asset]["linked_attached"] = self._check_for_linked_in_dict(asset,check_dict)
    
    def _data_asset_base_name(self, name):
        return name.split("_")[0] + "_" +  name.split("_")[1] + "_" + name.split("_")[2]


    def _check_for_linked_in_dict(self, name,check_dict):

        input_base_name = self._data_asset_base_name(name)
        for asset in check_dict:

            base_name = self._data_asset_base_name(asset)
            if input_base_name == base_name:
                return True
        return False


    def _get_id_for_name(self, name):
        for r in self.all_data_assets:
            if r["name"] == name:
                return r["id"]
        return None

    def get_attached_data_assets(self, data_dir = "/root/capsule/data/"):
        # get names of folders in the data dir
        data_dir = Path(data_dir)

        data_assets_dict = {}
        data_assets = [f for f in data_dir.iterdir() if f.is_dir()]

        names = [f.name for f in data_assets]

        for name in names:
            data_assets_dict[name] = {}
            data_assets_dict[name]["path"] = data_dir / name

            # asset type derived if name contains "processed", raw otherwise
            if "processed" in name:
                data_assets_dict[name]["type"] = "processed"
            else:
                data_assets_dict[name]["type"] = "raw"


            # set id
            data_assets_dict[name]["id"] = self._get_id_for_name(name)


            

        return data_assets_dict






    def attach_assets(data: Union[list, str, Path], api_secret_name = "CODEOCEAN_TOKEN"):
        """
        list should be dict with "name" and "id" fields
        
        "API_SECRET"
        """

        if isinstance(data, (str, Path)):
            with open(data, mode="r") as f:
                data = yaml.safe_load(f)
        elif isinstance(data, list):
            for d in data:
                assert isinstance(d, dict), "All elements of the list must be dictionaries"

        url = f'https://codeocean.allenneuraldynamics.org/api/v1/capsules/{os.getenv("CO_CAPSULE_ID")}/data_assets'
        headers = {"Content-Type": "application/json"} 
        auth = ("'" + os.getenv(api_secret_name), "'")
        print(data)
        response = requests.post(url=url, headers=headers, auth=auth, json=data)
        print(response.text)