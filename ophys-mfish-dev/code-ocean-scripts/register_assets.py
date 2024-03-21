"""
Use this script to register data assets in Code Ocean. The idea is to use a run_response json that we generate from "run_capsule_*.py" scripts to
get the computation ids of interest and register the results. We resave the run_response json indicated the data assets is regisetered.


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

# args
import argparse

parser = argparse.ArgumentParser(description="Run a capsule in Code Ocean")
parser.add_argument("--run_json", type=str, help="The run json to use", required=False)


# TODO: add way to check if already processed
def main(run_json,tags):
    co_cred = CodeOceanCredentials()
    co_client = CodeOceanClient.from_credentials(co_cred)

    data = []
    for run_dict in run_json:
        #da_name = co_client.get_data_asset(da_id).json()["name"]

        processed_name = run_dict['asset_name_processed'].replace('processed','dlc-eye')
        run_id = run_dict['id']

        register_request = CreateDataAssetRequest(
            mount=processed_name,
            tags=tags,
            source=Source(computation ={"id": run_id})
        )

        register_response = co_client.create_new_data_asset(register_request).json()

        print(f"Run response: {run_response}")
        time.sleep(5)

    # TODO: resave runjson in register asset json       
    #     processed_asset_name = da_name + "_processed_" + proc_time
    #     run_response["asset_name_processed"] = processed_asset_name
    #     run_response["asset_id"] = da_id
    #     run_response["asset_name"] = da_name
    #     data.append(run_response)

    # timestamp = dt.now().strftime("%Y%m%dT%H%M%S")
    # with open(f"run_results_{timestamp}.json", "w") as fp:
    #     json.dump(data, fp, indent=4)

if __name__ == "__main__":
     args = parser.parse_args()

    run_json = args.run_json
    run_json = "/ophys-mfish-dev/ophys-mfish-dev/code-ocean-scripts/run_jsons/run_results_20240315T212411.json"
    with open(run_json, "r") as f:
        run_json = json.load(f)


    tags = ["derived", "multiplane-ophys","dlc-eye"]
    main(run_json,tags = tags)

