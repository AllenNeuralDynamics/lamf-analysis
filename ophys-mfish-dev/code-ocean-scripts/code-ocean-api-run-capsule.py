from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_api.credentials import CodeOceanCredentials
import time
from datetime import datetime as dt
from datetime import timezone as tz
import json

# args
import argparse

parser = argparse.ArgumentParser(description="Run a capsule in Code Ocean")
parser.add_argument("--capsule_id", type=str, help="The capsule id to run")
parser.add_argument("--data_assets", type=str, nargs="+", help="The data assets to use")


def main(capsule_id, data_assets):
    co_cred = CodeOceanCredentials()
    print(co_cred)

    # Creading the API Client
    co_client = CodeOceanClient.from_credentials(co_cred)

    capsule_id = "36173b3e-2e7b-4510-ad51-8b7e90be08bc"

    datasets_ids = {
    "1304601605": "b62facdf-90dc-45d1-bb9f-b9fd269e0405",
    "1308185787": "ac0e5f8b-f2dd-4163-b4c8-ca8936b72e82",
    "1312420817":"d71eba24-36de-4707-9041-8d1f4e15139b",
    "1311805667": "f7687e92-efd4-4cba-ae3f-a111af1da6b3",
    "1311436357":"73117348-63cb-46b6-a1a3-e51b9bd8fe46"
       
    }

    data = []
    for identifier, data_asset_id in datasets_ids.items():
        print(f"Running dataset {identifier}")

        
        data_assets = [
            {"id": data_asset_id, "mount": "input"},
        ]

        run_response = co_client.run_capsule(
            capsule_id=capsule_id,
            data_assets=data_assets,
            parameters=None,  # []#[dumped_parameters],
        ).json()
        print(f"Run response: {run_response}")
        proc_time = dt.now(tz.utc).strftime("%Y-%m-%d_%H-%M-%S")
        time.sleep(5)

        data_asset = co_client.get_data_asset(data_asset_id).json()["name"]
        processed_asset_name = data_asset + "_processed_" + proc_time
        run_response["session"] = identifier
        run_response["asset_name_processed"] = processed_asset_name
        run_response["asset_id"] = data_asset_id
        data.append(run_response)
        time.sleep(30)
    timestamp = dt.now().strftime("%Y%m%dT%H%M%S")
    with open(f"run_results_{timestamp}.json", "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    # capsule_id = args.capsule_id
    # data_assets = args.data_assets
    cap

    print(data_assets)
    print(f"Running capsule {capsule_id} with data assets {data_assets}")

    

    main()
