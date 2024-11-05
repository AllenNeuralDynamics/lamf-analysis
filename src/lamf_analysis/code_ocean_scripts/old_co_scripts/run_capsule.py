# Description
# -----------
# The script will run a capsule with a list of data assets and save run results to json.
# Inputs: (1) command line, --capule_id and --data_assets or (2) hard-coded in the script (see example function)
# Outputs: json file with run results. This is necessary to for next step, to register the data assets.

# Usage
# -----
# + make sure code ocean credentials are avaible for aind_codeocean_api

# TODO:
# -----
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
from pathlib import Path



# not implemented yet
# parser.add_argument("--yml_path", type=str, help="The path to the yml file", required=False)
# parser.add_argument("--capsule_name", type=str, help="The name of the capsule", required=False)


def eye_tracking_gcamp_pilots_test():

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


def multiplane_ophys_pipeline_v3_tin():
    capsule_id = "543b69f3-cf29-47b7-8fda-c1482557e814"

    data_assets = ['89421805-ed71-487a-aac7-917cb9c1f899',
                   '8aa984c6-9fc5-465c-9bc7-373a232c8283',
                   '2a96c6c6-2f52-4f76-a18b-c3855be28874',
                   # '00e71d71-0a88-4ecd-ba81-6df542d120c0', # Duplicate 14
                   # '24876419-b80f-4ae2-9772-2be0c554e6c7' # Duplicate 17
                   ]

    data_assets = ['74de10a8-bc4a-4422-91de-ae496b0739fe']  # 17

    data_assets = ['02d3dedf-5722-438b-82cf-f957a0e9b220']  # 18

    data_assets = [
                   # '977de373-b163-4567-b17d-1f7141c7d502',  # Duplicate 18
                   # '833a6494-1a9f-4ac1-835e-841caad8dd74',  # Dupplicate 18
                #    'cf251b85-2444-4180-97b8-637629213494',
                #    'd80e7895-cbc6-46b6-9cf9-22a3ac75ae0f',
                #    'ab53c592-dbdb-40a8-95a4-54b64dbebcea',
                #    'd85dc090-9401-439c-b875-1a8dad5cdd5c',
                #    '94c31974-9352-47e0-ad9e-b68d53d174b9']
                   'd571f0e3-eef5-4409-941c-5e2bb9bb8c1f',
                   '64fe1c1c-fd62-48aa-97cb-4a5a10c53b92',
                   'ed117478-a443-482b-abe3-a8e1561b7d53',
                   '1db9390d-5c26-4a0d-b37a-20cbf8984b08',
                   '929a4fd0-ba9f-470c-adbe-8aada1c86ed3']
                #    'ff24db95-8407-4bba-af0c-bf255d066947',
                #    '65e0c68c-6d72-4927-a04b-6b50d4ad81cd']

    mount = "multiplane-ophys_457841_2019-09-26_10-40-09"

    return capsule_id, data_assets


def oi4_jun62024():

    data_assets = ['fc1cdfec-f058-412d-9ec6-8d511427ee7b', # AIND trigger ran 6/11
                    '5833ef57-4c70-4358-8d56-e3ca4da59bac', # AIND trigger ran 6/11
                    '944aa90d-575a-474c-8169-81648adfd9da', # AIND trigger ran 6/11
                    'ef8d91d8-8283-440b-9fd3-63a69295e141'] # AIND trigger ran 6/11
    
    #data_assets = ['350148c1-ccfc-4e9c-96a8-e43b8a3a2fc9'] # mounted asset

    # v5
    capsule_id = "56bf687b-dbcd-4b93-a650-21b8584036ff"
    mount = "multiplane-ophys_726433_2024-05-14_08-13-02"
    #tags = ["multiplane-ophys", "ophys-mfish", "gcamp-validation", 'file-splitting']

    return capsule_id, data_assets, mount

def run_capsule(capsule_id, data_asset_ids, mount, dry_run=False):
    co_cred = CodeOceanCredentials()
    if co_cred is None:  # might not need check if class handles
        raise ValueError("No credentials found")
    else:
        print(f"Using credentials: {co_cred}")
    co_client = CodeOceanClient.from_credentials(co_cred)

    data = []
    for da_id in data_asset_ids:
        da_name = co_client.get_data_asset(da_id).json()["name"]

        data_assets = [ComputationDataAsset(
            id=da_id,
            # mount=da_name,
            mount=mount  # might get computation id results; for multiplane pipeline; this is the multiplane mount
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
        run_response["derived_asset_name"] = processed_asset_name
        run_response["asset_id"] = da_id
        run_response["asset_name"] = da_name
        data.append(run_response)
        time.sleep(30)

    # save run results to json
    if not dry_run:
        timestamp = dt.now().strftime("%Y%m%dT%H%M%S")
        run_json_path = Path("/allen/programs/mindscope/workgroups/learning/code_ocean/run_jsons")
        json_path = run_json_path / f"run_results_{timestamp}.json"
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a capsule in Code Ocean")
    parser.add_argument("--capsule_id", type=str, help="The capsule id to run", required=False)
    parser.add_argument("--data_assets", type=str, nargs="+", help="The data assets to use", required=False)
    parser.add_argument("--dry_run", action="store_true", help="Dry run", required=False)

    args = parser.parse_args()
    capsule_id = args.capsule_id
    data_assets = args.data_assets
    dry_run = args.dry_run

    if capsule_id is None or data_assets is None:
        #capsule_id, data_assets = example_define_inputs()
        capsule_id, data_assets, mount = oi4_jun62024()

    print(f"Running capsule {capsule_id} with data assets {data_assets}")
    run_capsule(capsule_id, data_assets, mount, dry_run)
