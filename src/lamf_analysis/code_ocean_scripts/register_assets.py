""" NOTE: not finished (3/21/2024)
Use this script to register data assets in Code Ocean. The idea is to use a run_response json that we generate from "run_capsule.py" scripts to
get the computation ids of interest and register the results. We resave the run_response json indicated the data assets is registered.


The processed name needs to altered depending on the data type:
+ processed
+ dlc-eye
+ dlc-face
+ dlc-body

Tags for data types:
+ processed:
+ dlc-eye: ["derived", "multiplane-ophys","dlc-eye"]

"""


from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_api.credentials import CodeOceanCredentials
from aind_codeocean_api.models.data_assets_requests import CreateDataAssetRequest
from aind_codeocean_api.models.data_assets_requests import Source

import time
from datetime import datetime as dt
from datetime import timezone as tz
import json
from pathlib import Path
import yaml
# args
import argparse

parser = argparse.ArgumentParser(description="Run a capsule in Code Ocean")
parser.add_argument("--run_json", type=str, help="The run json to use", required=False)


def get_capsule_tags(capsule_name, yml_path=Path("./useful_capsules.yml")):
    with open(yml_path, "r") as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        for capsule in yml:
            if capsule_name == capsule['name']:
                tags = capsule["processed_tags"]
                capsule_id= capsule["capsule_id"]
            else:
                raise ValueError(f"No capsule found with name {capsule_name} in {yml_path}")
    return tags


def register_assets(run_json_path, capsule_name):
    co_cred = CodeOceanCredentials()
    co_client = CodeOceanClient.from_credentials(co_cred)

    with open(run_json_path, "r") as f:
        run_json = json.load(f)

    tags = get_capsule_tags(capsule_name)

    new_json_dicts = []
    for run_dict in run_json:
        #da_name = co_client.get_data_asset(da_id).json()["name"]

        processed_name = run_dict['asset_name_processed'].replace('processed', 'dlc-eye')
        run_id = run_dict['id']

        register_request = CreateDataAssetRequest(
            name=processed_name,
            mount=processed_name,
            tags=tags,
            source=Source(computation ={"id": run_id})
        )

        # # actually register
        # register_response = co_client.create_data_asset(register_request).json()
        # print(f"Run response: {register_response}")
        # time.sleep(5)

        run_dict['registered'] = True
        run_dict['asset_name_processed'] = processed_name
        run_dict['asset_tags_processed'] = tags
        new_json_dicts.append(run_dict)

    # resave run_json as *_registered.json
    run_json_path = run_json_path.replace(".json", "_registered.json")
    with open(run_json_path, "w") as f:
        json.dump(new_json_dicts, f, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    run_json_path = args.run_json
    run_json_path = "/ophys-mfish-dev/ophys-mfish-dev/code-ocean-scripts/run_jsons/run_results_20240315T210715.json"
    run_json_path = "/ophys-mfish-dev/ophys-mfish-dev/code-ocean-scripts/run_jsons/run_results_20240315T212411.json"

    # WARNING:
    capsule_name = "aind-capsule-eye-tracking"
    capsule_name = "Multiplane Ophys Pipeline - with retries"

    register_assets(run_json_path, capsule_name)

