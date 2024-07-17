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
    def __init__(self,
                 query: Optional[str] = "multiplane",
                 CO_TOKEN_VAR_NAME: Optional[str] = "CODEOCEAN_TOKEN"):
        super().__init__(query=query)

        self.CO_TOKEN_VAR_NAME = CO_TOKEN_VAR_NAME

        # assets
        #self.assets = self.get_attached_data_assets()

        # adds "linked_attached" key; 
        # bool that indicates in linked (raw/processed) data asset is attached
        #self._update_linked_key()
        
        # MJD 07/14/2024
        print("DataAssetLoader under construction; main function to use is dal.attach_assets() or dal.attach_assets_in_df()")

    def attach_assets(self, 
                      assets: Union[list, str, Path],
                      print_response: bool = False):
        """Attach data assets to the capsule

        Parameters
        ----------
        assets : Union[list, str, Path]
            Data assets to attach to the capsule. If str or Path, should be a path to a yaml file.
            list: list of dictionaries with "name" and "id" fields for each data asset,
        api_secret_name : str
            Name of the environment variable containing the API secret
        print_response : bool
            Print the response from the API call

        Returns
        -------
        None
        """

        if isinstance(assets, (str, Path)):
            with open(assets, mode="r") as f:
                assets = yaml.safe_load(f)
        elif isinstance(assets, list):
            for d in assets:
                assert isinstance(d, dict), "All elements of the list must be dictionaries"

        url = f'https://codeocean.allenneuraldynamics.org/api/v1/capsules/{os.getenv("CO_CAPSULE_ID")}/data_assets'
        headers = {"Content-Type": "application/json"} 
        auth = ("'" + os.getenv(self.CO_TOKEN_VAR_NAME), "'")

        responses = requests.post(url=url, headers=headers, auth=auth, json=assets).json()
        print(responses)

        # iterate reponse.text

        for response in responses:
            asset_id = response['id']
            if response["mount_state"] =="unchanged":
                print(f"Data asset {asset_id} not mounted (already attached)")
            elif response["mount_state"] =="mounted":
                print(f"Data asset {asset_id} attached!")
        if print_response:
            print(response.text)

    def attach_assets_in_df(self,
                            df: pd.DataFrame,
                            print_response: Optional[bool] = False) -> None:
        """Attach data assets to the capsule from a dataframe

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns "raw_data_asset_id", "raw_data_asset_name", "data_asset_id", "data_asset_name"
        print_response : bool
            Print the response from the API call

        Returns
        -------
        None
        """
        assets = []
        for i, asset_row in df.iterrows():
            linked_assets = self._linked_asset_from_row(asset_row)
            assets.extend(linked_assets)

        self.attach_assets(assets=assets, print_response=print_response)

    def _linked_asset_from_row(self,
                               asset_row: pd.Series) -> list:
        """Return a list of linked assets from a row of a dataframe

        Parameters
        ----------
        asset_row : pd.Series
            Row of a dataframe with columns "raw_data_asset_id", "raw_data_asset_name", "data_asset_id", "data_asset_name"

        Returns
        -------
        list
            List of dictionaries with keys "id" and "name" for each linked asset
        """
        linked_assets = [{"id": asset_row["raw_data_asset_id"], "name": asset_row["raw_data_asset_name"]},
                        {"id": asset_row["data_asset_id"], "name": asset_row["data_asset_name"]}]
        return linked_assets


    def attach(self, 
               data: Union[dict,list,Path]):
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


    # def attach_linked_data_asset(self, 
    #                             name: Optional[str] = None, 
    #                             id: str = Optional[None], 
    #                             data_dir = "/root/capsule/data/"):
    #     """Attach a linked data asset to the capsule

    #     A lot of data assets come in pairs, raw and derived/processed, these are considered "linked" assets.
    #     This method was developed for Multiplane-ophys experiments, which have
    #     a particular naming schema (see class docstring for examples).

    #     raw: 'multiplane-ophys_677594_2023-08-04_09-44-08'
    #     Processed: 'multiplane-ophys_677594_2023-08-04_09-44-08_processed_2024-02-08_23-26-44'

    #     The raw name is considered the "basename". To find the linked asset, look for the matching basename
    #     in in the list of available assets, which is retrieved by the OceanCodeDataExplorer class.

    #     This class is bidirectional, raw or processed can be attached first.


    #     Parameters
    #     ----------
    #     name : str
    #         Name of data asset to attach
    #     id : str
    #         ID of data asset to attach
    #     data_dir : str
    #         Directory containing data assets
    #     """

    #     try :
    #         if name is not None:
    #             id = self.assets[name]["id"]
    #         elif id is not None:
    #             name = [k for k,v in self.assets.items() if v["id"] == id][0]
    #         else:
    #             raise ValueError("Must provide either name or id")

    #         data = {"name": name, "id": id}
    #     except KeyError:
    #         print(f"Data asset {name} not found in attached data assets")

    #         # TODO: could attach if not already attached
    #         return

    #     # determine if asset is raw or processed, look for opposite 
    #     derived = True if "processed" in name else False
    #     if derived:
    #         df = pd.DataFrame(self.raw_assets)
    #     else:
    #         df = pd.DataFrame(self.derived_assets)

    #     # use basename to match across linked assets
    #     df["basename"] = df["name"].apply(self._data_asset_base_name)
    #     linked_name = df[df["basename"] == self._data_asset_base_name(name)]["name"].values[0]
    #     linked_id = df[df["basename"] == self._data_asset_base_name(name)]["id"].values[0]
    #     print(linked_name, linked_id)

    #     if linked_name is None:
    #         print(f"LINKED ASSET NOT FOUND: {name}")
    #         return

    #     if self._check_if_attached(linked_name):
    #         print(f"LINKED ASSET ALREADY ATTACHED: {name}")
    #         return
    #     else:
    #         data = {"name": linked_name, "id": linked_id}
    #         self.attach(data)
        
    #     return df

    # @property
    # def linked_data_assets(self):
    #     return {k:v for k,v in self.assets.items() if v["linked_attached"]}

    # @property
    # def unlinked_data_assets(self):
    #     return {k:v for k,v in self.assets.items() if not v["linked_attached"]}

    # def _update_linked_key(self):
        
    #     for asset in self.assets:
    #         check_dict = self.assets.copy()
    #         check_dict.pop(asset)
    #         # remove asset from check dict

    #         self.assets[asset]["linked_attached"] = self._check_for_linked_in_dict(asset,check_dict)
    
    # def _data_asset_base_name(self, name):
    #     return name.split("_")[0] + "_" +  name.split("_")[1] + "_" + name.split("_")[2]

    # def _check_if_attached(self, name):
    #     for asset in self.assets:
    #         if name == asset:
    #             return True
    #     return False


    # def _check_for_linked_in_dict(self, name,check_dict):
    #     try: 
    #         input_base_name = self._data_asset_base_name(name)
    #         for asset in check_dict:

    #             base_name = self._data_asset_base_name(asset)
    #             if input_base_name == base_name:
    #                 return True
    #     except IndexError:
    #         # likely not aind session name form
    #         pass
    #     return False

    def _get_id_for_name(self, name):
        for r in self.result:
            if r["name"] == name:
                return r["id"]
        return None

    # def get_attached_data_assets(self, data_dir = "/root/capsule/data/"):
    #     # get names of folders in the data dir
    #     data_dir = Path(data_dir)

    #     data_assets_dict = {}
    #     data_assets = [f for f in data_dir.iterdir() if f.is_dir()]

    #     names = [f.name for f in data_assets]

    #     for name in names:
    #         data_assets_dict[name] = {}
    #         data_assets_dict[name]["path"] = data_dir / name

    #         # asset type derived if name contains "processed", raw otherwise
    #         if "processed" in name:
    #             data_assets_dict[name]["type"] = "processed"
    #         else:
    #             data_assets_dict[name]["type"] = "raw"


    #         # set id
    #         data_assets_dict[name]["id"] = self._get_id_for_name(name)

    #     return data_assets_dict
