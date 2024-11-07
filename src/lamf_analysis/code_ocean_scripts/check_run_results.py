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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a capsule in Code Ocean")
    parser.add_argument("--computation_id", type=str, help="", required=False)

    computation_id = args.computation_id

    if computation_id is None:
        computation_id = "89255913-beb9-4f9b-bdd2-b64a662de2c9"

    co_cred = CodeOceanCredentials()
    co_client = CodeOceanClient.from_credentials(co_cred)

    co_client.get_capsule_computations(computation_id)

    args = parser.parse_args()