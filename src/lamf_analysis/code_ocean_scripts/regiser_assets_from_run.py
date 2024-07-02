# %%
from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_api.models.computations_requests import RunCapsuleRequest
from aind_codeocean_api.models.computations_requests import ComputationDataAsset
import os
import time
from datetime import datetime as dt
from datetime import timezone as tz

from aind_codeocean_api.models.data_assets_requests import CreateDataAssetRequest
from aind_codeocean_api.models.data_assets_requests import Source, Target

co_client = CodeOceanClient(os.getenv("CODEOCEAN_DOMAIN"), os.getenv("CODEOCEAN_TOKEN"))

capsule_id = "56bf687b-dbcd-4b93-a650-21b8584036ff" # V5
#capsule_id = "cd3897a2-bc35-4d9a-9bf2-d86fd9c850ab" # V4
results = co_client.get_capsule_computations(capsule_id)

tags = ['ophys-mfish','gcamp-validation','derived', 
        'multiplane-ophys','pipeline-v5.0']
# for run_id in [7630384, 7630420, 7630456, 7630529]: # running
#for run_id in [7775457, 7774948, 7723998]: # Asset, 
for run_id in [7775457]:
    for i in results.json():
        run_str = f'Run {run_id}'
        if i["name"] == run_str:
            print(i)
            comp_id = i['id']
            da_id = i['data_assets'][0]['id']  # NOTE [0] may not akways be true, multiple data assets possible
            mount = i['data_assets'][0]['mount']

            # get data asset
            data_asset = co_client.get_data_asset(da_id).json()
            proc_time = dt.now(tz.utc).strftime("%Y-%m-%d_%H-%M-%S")
            processed_name = data_asset['name'] + "_processed_" + proc_time

            # info
            print(f"Run ID: {run_id}")
            print(f"processed name: {processed_name}")
            print(f"\n")
            # print(comp_id, processed_name, mount)

            # create data asset
            dry_run = False
            if not dry_run:
                register_request = CreateDataAssetRequest(
                    name=processed_name,
                    mount=processed_name,
                    tags=tags,
                    source=Source(computation={"id": comp_id}),
                )
                run_results = co_client.create_data_asset(register_request)
                print(run_results.json())
                time.sleep(5)


# %%
from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_api.models.computations_requests import RunCapsuleRequest
from aind_codeocean_api.models.computations_requests import ComputationDataAsset
import os
import time

co_client = CodeOceanClient(os.getenv("CODEOCEAN_DOMAIN"), os.getenv("CODEOCEAN_TOKEN"))
results = co_client.get_capsule_computations("56bf687b-dbcd-4b93-a650-21b8584036ff")


for i in results.json():
    print(i['name'])
    if i['name'] == 'Run 7630420':
        print(i)
        comp_id = i['id']
        da_id = i['data_assets'][0]['id']  # NOTE [0] may not akways be true, multiple data assets possible
        mount = i['data_assets'][0]['mount']

        data_assets = [ComputationDataAsset(
            id=da_id,
            mount=mount
        )]

        print(comp_id, data_assets, mount)

        capsule_request = RunCapsuleRequest(
                capsule_id = capsule_id,
                resume_run_id=comp_id,
                data_assets=data_assets,
                )
        return_results = co_client.run_capsule(capsule_request)
        print(return_results.json())
        time.sleep(30)
