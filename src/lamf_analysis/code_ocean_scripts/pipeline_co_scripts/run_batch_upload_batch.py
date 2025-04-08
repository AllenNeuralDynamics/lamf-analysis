"""
This script is used to upload a batch of data assets to Code Ocean.

Usage:
python run_batch_upload_batch.py --csv_file <path> --pipeline_id <id> --max-jobs 10
csv files must contain batch, subject_id, mount, and asset_id or asset_name
if using the asset_name, add flag --asset_name True

Real example:
python run_batch_upload_batch.py --csv_file /home/matt.davis/code/mdev/src/mdev/batch_process_capsule_csvs/2025-02-13_dlc-eye_oi1_2.csv 
--pipeline_id 4cf0be83-2245-4bb1-a55c-a78201b14bfe --process_name_suffix dlc-eye --max-jobs 20

Example csv file:
batch,asset_id,mount,subject_id
1,54e423fc-f898-4b31-bfb2-3ce2dea597e2,multiplane-ophys_721291_2024-05-16_08-57-00,721291
1,0a48800d-9fe8-4c49-aeb6-e6cc244cda1b,multiplane-ophys_721291_2024-05-17_08-35-11,721291
2,f9481739-65de-4594-af87-f6e610f02364,multiplane-ophys_721291_2024-05-18_08-55-42,721291
2,851d7348-eaac-47a3-88e4-a56f3c47b9c5,multiplane-ophys_739564_2024-09-17_12-13-04,739564
3,88c42414-142a-46fc-825f-528fa4834aaf,multiplane-ophys_739564_2024-09-19_14-27-30,739564

This will submit 3 jobs to Code Ocean, one for each batch of data assets.
"""

import argparse
import csv
import logging
import os
import time
import json
from dataclasses import dataclass
from typing import Union, List, Dict
from collections import defaultdict

from aind_codeocean_pipeline_monitor.models import (CaptureSettings,
                                                    PipelineMonitorSettings)
from aind_data_access_api.document_db import MetadataDbClient
from codeocean import CodeOcean
from codeocean.computation import (ComputationState, DataAssetsRunParam,
                                   RunParams)
from dataclasses_json import dataclass_json

logging.basicConfig(
    filename="batch.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Set environment variables
API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"
domain = os.getenv("CODEOCEAN_DOMAIN")
token = os.getenv("CODEOCEAN_TOKEN")

#monitor_pipeline_capsule_id = os.getenv("CO_MONITOR_PIPELINE")
monitor_pipeline_capsule_id = "fbdddd96-6d2a-4e40-88c8-45f3f84cfaf3" # og
client = CodeOcean(domain=domain, token=token)


@dataclass_json
@dataclass(frozen=True)
class JobSettings:
    pipeline_id: str
    assets_list: List[Dict[str, str]]
    subject_id: str
    process_name_suffix: str

def get_monitor_settings(
    job_settings: Union[JobSettings, dict],
) -> PipelineMonitorSettings:
    """Get the pipeline monitor settings.
    Parameters
    ----------
    job_setting: Union[JobSettings, dict]
        Settings defining the job to run
    Returns
    -------
    PipelineMonitorSettings
        The pipeline monitor settings.
    """
    if isinstance(job_settings, dict):
        job_settings = JobSettings.from_dict(job_settings)
    return PipelineMonitorSettings(
        run_params=RunParams(
            capsule_id=job_settings.pipeline_id,
            data_assets=[
                DataAssetsRunParam(
                    id=asset["id"],
                    mount=asset["mount"],
                ) for asset in job_settings.assets_list
            ],
        ),
        capture_settings=CaptureSettings(
            process_name_suffix=job_settings.process_name_suffix,
            tags=["derived", "multiplane-ophys", job_settings.subject_id],
            custom_metadata={
                "data level": "derived",
                "experiment type": "multiplane-ophys",
                "subject id": job_settings.subject_id,
            },
        ),
    )


def get_asset_id(docdb_api_client, asset_name) -> str:
    """Get the asset ID from the data access api
    Parameters
    ----------
    docdb_api_client : MetadataDbClient
        The data access api client
    asset_name : str
        The asset name
    Returns
    -------
    str
        The asset ID
    """
    query = {"name": asset_name}
    projection = {"external_links": 1}
    response = docdb_api_client.retrieve_docdb_records(
        filter_query=query, projection=projection
    )
    external_links = response[0].get("external_links", None)
    if type(external_links) is str:
        external_links = json.loads(external_links)
        external_links = external_links.get("Code Ocean", None)
    if type(external_links) is list and len(external_links) > 1:
        external_links = external_links[0]
        external_links = external_links.get("Code Ocean", None)
    if type(external_links) is dict:
        try:
            external_links = external_links.get("Code Ocean", None)[0]
        except IndexError:
            external_links = "None"
    if type(external_links) is list:
        try:
            external_links = external_links[0]
        except IndexError:
            external_links = "None"
    return external_links


def run():
    """Example usage below
    > python batch_process.py --csv_file <path> --pipeline_id <id> --max-jobs 10
    csv files must contain batch, subject_id, mount, and asset_id or asset_name
    if using the asset_name, add flag --asset_name True
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--pipeline_id", type=str, required=True)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--sleep", type=int, default=600)
    parser.add_argument("--asset_name", type=bool, default=False)
    parser.add_argument("--process_name_suffix", type=str, default="processed")
    args = parser.parse_args()
    rows = []
    asset_name = args.asset_name

    with open(args.csv_file, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            rows.append(row)
    header = rows[0]
    header = [i.strip() for i in header]
    data = rows[1:]

    # Group data by batch
    batch_groups = defaultdict(list)
    docdb_api_client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )

    if asset_name:
        asset_id_index = header.index("asset_name")
    else:
        asset_id_index = header.index("asset_id")
    batch_index = header.index("batch")
    subject_id_index = header.index("subject_id")
    mount_index = header.index("mount")

    # Group assets by batch
    for row in data:
        if not row:
            continue
        batch_num = row[batch_index]
        if asset_name:
            data_asset_id = get_asset_id(docdb_api_client, row[asset_id_index])
        else:
            data_asset_id = row[asset_id_index]

        batch_groups[batch_num].append({
            "id": data_asset_id,
            "mount": row[mount_index],
            "subject_id": row[subject_id_index]
        })

    jobs = []
    # Process each batch as a single job
    for batch_num, assets in batch_groups.items():
        if not assets:
            continue

        # Use the first subject_id in the batch for metadata
        subject_id = assets[0]["subject_id"]
        
        job_settings = JobSettings(
            pipeline_id=args.pipeline_id,
            assets_list=[{"id": asset["id"], "mount": asset["mount"]} 
                        for asset in assets],
            subject_id=subject_id,
            process_name_suffix=args.process_name_suffix,
        )

        settings = get_monitor_settings(job_settings)
        pipeline_params = settings.model_dump_json(exclude_none=True)
        monitor_params = RunParams(
            capsule_id=monitor_pipeline_capsule_id,
            parameters=[pipeline_params]
        )
        monitor_run_comp = client.computations.run_capsule(monitor_params)
        job_id = monitor_run_comp.id
        jobs.append(monitor_run_comp)
        logging.info(f"Batch {batch_num} job {job_id} started")
        print(f"Jobs started: {len(jobs)}")

        while len(jobs) >= args.max_jobs:
            for job in jobs:
                state = client.computations.get_computation(job.id).state
                if state in [ComputationState.Completed,
                           ComputationState.Failed]:
                    if state == ComputationState.Failed:
                        logging.error(f"Job {job.id} failed")
                    jobs.remove(job)
                    break
            time.sleep(args.sleep)


if __name__ == "__main__":
    run()