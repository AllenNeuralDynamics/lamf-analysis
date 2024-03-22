from .code_ocean_data_explorer import CodeOceanDataExplorer

import os
import yaml
import requests
from typing import Optional, Union
from pathlib import Path
import pandas as pd


class DataAssetLoader(CodeOceanDataExplorer):
    """Organize and attach data assets to a capsule

    Developed for Multiplane-ophys experiments.

    E.g. Data assets
    raw: 'multiplane-ophys_677594_2023-08-04_09-44-08'
    Processed: 'multiplane-ophys_677594_2023-08-04_09-44-08_processed_2024-02-08_23-26-44'


    """
    def __init__(self):
        super().__init__()

        # assets
        self.assets = self.get_attached_data_assets()

        # adds "linked_attached" key; 
        # bool that indicates in linked (raw/processed) data asset is attached
        self._update_linked_key()


    def attach(self, 
               data: Union[dict,list, Path]):
        """Attach a data asset to the capsule and update DAL


        Parameters
        ----------
        data : Union[dict,list]
            Data asset to attach to the capsule. If list, should be a list of dictionaries with 
            at minimum "name" and "id" fields.

        """
        if isinstance(data, dict):
            data = [data]

        self.attach_assets_to_capsule(data=data) # list
        self.assets = self.get_attached_data_assets()
        self._update_linked_key()


    def attach_linked_data_asset(self, 
                                name: Optional[str] = None, 
                                id: str = Optional[None], 
                                data_dir = "/root/capsule/data/"):
        """Attach a linked data asset to the capsule

        A lot of data assets come in pairs, raw and derived/processed, these are considered "linked" assets.
        This method was developed for Multiplane-ophys experiments, which have
        a particular naming schema (see class docstring for examples).

        raw: 'multiplane-ophys_677594_2023-08-04_09-44-08'
        Processed: 'multiplane-ophys_677594_2023-08-04_09-44-08_processed_2024-02-08_23-26-44'

        The raw name is considered the "basename". To find the linked asset, look for the matching basename
        in in the list of available assets, which is retrieved by the OceanCodeDataExplorer class.

        This class is bidirectional, raw or processed can be attached first.


        Parameters
        ----------
        name : str
            Name of data asset to attach
        id : str
            ID of data asset to attach
        data_dir : str
            Directory containing data assets
        """

        try :
            if name is not None:
                id = self.assets[name]["id"]
            elif id is not None:
                name = [k for k,v in self.assets.items() if v["id"] == id][0]
            else:
                raise ValueError("Must provide either name or id")

            data = {"name": name, "id": id}
        except KeyError:
            print(f"Data asset {name} not found in attached data assets")

            # TODO: could attach if not already attached
            return
        # Could also look at tags
        derived = True if len(name.split("_")) > 3 else False
        if derived:
            df = pd.DataFrame(self.raw_assets)
        else:
            df = pd.DataFrame(self.derived_assets)

        # use basename to match across linked assets
        df["basename"] = df["name"].apply(self._data_asset_base_name)
        linked_name = df[df["basename"] == self._data_asset_base_name(name)]["name"].values[0]
        linked_id = df[df["basename"] == self._data_asset_base_name(name)]["id"].values[0]
        print(linked_name, linked_id)

        if linked_name is None:
            print(f"LINKED ASSET NOT FOUND: {name}")
            return

        if self._check_if_attached(linked_name):
            print(f"LINKED ASSET ALREADY ATTACHED: {name}")
            return
        else:
            data = {"name": linked_name, "id": linked_id}
            self.attach(data)
        
        return df

    def attach_assets_to_capsule(self, data: Union[list, str, Path], api_secret_name = "CODEOCEAN_TOKEN"):
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

        response = requests.post(url=url, headers=headers, auth=auth, json=data)
        print(response.text)

        # TODO: handle response printing
        # LIST HANDLING
        # if response["mount_state "] =="unchanged":
        #     print(f"Data asset {data['name']} already attached")
        # elif response["mount_state "] =="mounted":
        #     print(f"Data asset {data['name']} attached")
        
    @property
    def linked_data_assets(self):
        return {k:v for k,v in self.assets.items() if v["linked_attached"]}

    @property
    def unlinked_data_assets(self):
        return {k:v for k,v in self.assets.items() if not v["linked_attached"]}

    def _update_linked_key(self):
        
        for asset in self.assets:
            check_dict = self.assets.copy()
            check_dict.pop(asset)
            # remove asset from check dict

            self.assets[asset]["linked_attached"] = self._check_for_linked_in_dict(asset,check_dict)
    
    def _data_asset_base_name(self, name):
        try:
            base_name = name.split("_")[0] + "_" +  name.split("_")[1] + "_" + name.split("_")[2]
        except IndexError:
            base_name = name
        return 

    def _check_if_attached(self, name):
        for asset in self.assets:
            if name == asset:
                return True
        return False


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
            if "dlc-eye" in name:
                data_assets_dict[name]["type"] = "dlc-eye"
            else:
                data_assets_dict[name]["type"] = "raw"


            # set id
            data_assets_dict[name]["id"] = self._get_id_for_name(name)

        return data_assets_dict

    