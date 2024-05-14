# The script will run a capsule with a list of data assets and save run results to json.
# Inputs: (1) command line, --capule_id and --data_assets or (2) hard-coded in the script (see example function)
# Outputs: json file with run results. This is necessary to for next step, to register the data assets.

# Set-up:
# + Use/create a capule with the repo `aind_codeocean_api`
# + add code ocean credentials to your capsule

# TODO:
# - add way to check if already processed (a particular data asset and capsule id in run json)
# - load in yaml file for tags and capsule ids by name

from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_api.credentials import CodeOceanCredentials
from aind_codeocean_api.models.computations_requests import RunCapsuleRequest
from aind_codeocean_api.models.computations_requests import ComputationDataAsset

import time
from datetime import datetime as dt
from datetime import timezone as tz
import json
import argparse

parser = argparse.ArgumentParser(description="Run a capsule in Code Ocean")
parser.add_argument("--capsule_id", type=str, help="The capsule id to run", required=False)
parser.add_argument("--data_assets", type=str, nargs="+", help="The data assets to use", required=False)
parser.add_argument("--dry_run", action="store_true", help="Dry run", required=False)

# not implemented yet
# parser.add_argument("--yml_path", type=str, help="The path to the yml file", required=False)
# parser.add_argument("--capsule_name", type=str, help="The name of the capsule", required=False)

def example_define_inputs():

    # JK gcamp pilots 03/15/2024
    capsule_id = "4cf0be83-2245-4bb1-a55c-a78201b14bfe" # aind-capsule-eye-tracking

    data_assets = ["bc4013de-f522-419b-8525-54fe1513b813",
                   "71191107-40aa-4060-8375-d685a0d5c207",
                   "c65a4625-8b3b-4d3a-b300-a6f659cbcc3b",
                   "cf2ec3c9-9f93-419e-ae81-d2775de8acf4",
                   "2125ecb8-6569-40c5-ae61-414e76699b74",
                   "3b864edd-fec0-4a9d-9c4e-75189e260de7",
                   "1ca0ad6e-56d2-4bfa-bb46-55a276ca0287",
                   "b96ab4bd-f7a7-4dc2-9a0f-67a10823955c",
                   "099599a1-36f9-428d-9ac0-b9eb42cc522c",
                   "f545fb4e-7a07-429c-abce-aa9c4d4b9f42"]


    return capsule_id, data_assets


def multiplane_ophys_pipeline_v3():
    capsule_id = "543b69f3-cf29-47b7-8fda-c1482557e814"

    data_assets = ["082a3420-5cdf-45ea-8cc7-51ae3a111186"]

    return capsule_id, data_assets



def run_capsule(capsule_id, data_asset_ids, dry_run=False):
    co_cred = CodeOceanCredentials()
    if co_cred is None: # might not need check if class handles
        raise ValueError("No credentials found")
    else:
        print(f"Using credentials: {co_cred}")
    co_client = CodeOceanClient.from_credentials(co_cred)

    data = []
    for da_id in data_asset_ids:
        da_name = co_client.get_data_asset(da_id).json()["name"]

        
        data_assets = [ComputationDataAsset(
            id=da_id,
            mount=da_name,
        )]

        run_request = RunCapsuleRequest(
            capsule_id=capsule_id,
            data_assets=data_assets,
        )

        if dry_run:
            print(f"DRY RUN: {run_request}")
            continue

        run_response = co_client.run_capsule(run_request).json()
        print(f"Running dataset {da_name}")
        print(f"Run response: {run_response}")
        proc_time = dt.now(tz.utc).strftime("%Y-%m-%d_%H-%M-%S")
        time.sleep(5)

        
        processed_asset_name = da_name + "_processed_" + proc_time
        run_response["asset_name_processed"] = processed_asset_name
        run_response["asset_id"] = da_id
        run_response["asset_name"] = da_name
        data.append(run_response)
        time.sleep(30)

    if not dry_run:
        timestamp = dt.now().strftime("%Y%m%dT%H%M%S")
        with open(f"./run_jsons/run_results_{timestamp}.json", "w") as fp:
            json.dump(data, fp, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    capsule_id = args.capsule_id
    data_assets = args.data_assets
    dry_run = args.dry_run

    if capsule_id is None or data_assets is None:
        #capsule_id, data_assets = example_define_inputs()
        capsule_id, data_assets = multiplane_ophys_pipeline_v3()

    print(f"Running capsule {capsule_id} with data assets {data_assets}")
    run_capsule(capsule_id, data_assets, dry_run)
