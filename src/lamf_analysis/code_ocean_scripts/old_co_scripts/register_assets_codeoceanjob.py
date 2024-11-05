# %%
import os
from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_utils.codeocean_job import (
    CodeOceanJob, CodeOceanJobConfig
)

co_client = CodeOceanClient(os.getenv("CODEOCEAN_DOMAIN"), os.getenv("CODEOCEAN_TOKEN"))





tags = ["derived", "eye_tracking", "ophys-mfish"]
capsule_id ="4cf0be83-2245-4bb1-a55c-a78201b14bfe"
data_asset_id = "86cff6d2-cc50-4b41-b4ed-55f9ecefba94"
data_asset_mount = "multiplane-ophys_721291_2024-04-16_08-21-40"

data_assets = [dict(
                    id=data_asset_id,
                    #mount=data_asset_mount
                    )]

# Define Job Parameters
job_config_dict = dict(
    run_capsule_config = dict(
        data_assets=data_assets, # when None, the newly registered asset will be used
        capsule_id=capsule_id,
    #    input_data_mount=input_data_mount,
        run_parameters=[]
    ),
    capture_result_config = dict(
        process_name="dlc-eye",
        tags=tags # additional tags to the ones inherited from input
    )
)

# instantiate config model
job_config = CodeOceanJobConfig(**job_config_dict)

# instantiate code ocean job
co_job = CodeOceanJob(co_client=co_client, job_config=job_config)

# run and wait for results
job_response = co_job.run_job()
print(job_response)
# %%

tags = ['ophys-mfish','gcamp-validation','derived', 
        'multiplane-ophys','pipeline-v5.0']


data_asset_mount = 'multiplane-ophys_724567_2024-05-20_12-00-21'
data_asset_id = 'fc1cdfec-f058-412d-9ec6-8d511427ee7b'

capsule_id="56bf687b-dbcd-4b93-a650-21b8584036ff"
capsule_id = "791f711f-ff66-4818-ac4c-d1f177027113" # v5
input_data_mount="multiplane-ophys_726433_2024-05-14_08-13-02"