# using pydantic serialization inherent to PipelineMonitorSettings
# added named parameters as list of NamedRunParam
# batches are defined in the config file as a list of settings
#
# Example config:
# {
#     "settings_list": [
#         {
#             "capsule_id": "capsule1",
#             "tags": ["tag1", "tag2"],
#             "process_name_suffix": "suffix1",
#             "data_assets": [{"id": "asset1", "mount": "mount1"}],
#             "named_parameters": {"param1": "value1"}
#         },
#         {
#             "capsule_id": "capsule2",
#             "tags": ["tag3", "tag4"],
#             "process_name_suffix": "suffix2",
#             "data_assets": [{"id": "asset2", "mount": "mount2"}],
#             "named_parameters": {"param2": "value2"}
#         }
#     ]
# }

from aind_codeocean_pipeline_monitor.job import PipelineMonitorJob
from aind_codeocean_pipeline_monitor.models import (
    CaptureSettings,
    PipelineMonitorSettings,
)
from codeocean.capsule import Capsules
from codeocean.data_asset import DataAssets
from codeocean.computation import Computations, DataAssetsRunParam, RunParams, NamedRunParam
from codeocean import CodeOcean
import argparse
import logging
import time
import json
import os
import sys
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
import multiprocessing
from multiprocessing import Pool
from typing import List


def setup_logging():
    """Configure logging with timestamp in filename"""
    log_file = f"{os.path.basename(__file__)}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # Configure both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 'a' for append mode
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging to {log_file}")
    return log_file


def setup_codeocean_client():
    """Initialize and configure CodeOcean client with retry logic"""
    domain = os.getenv("CODEOCEAN_DOMAIN")
    token = os.getenv("CODEOCEAN_TOKEN")

    if not domain or not token:
        raise ValueError("CODEOCEAN_DOMAIN and CODEOCEAN_TOKEN environment variables must be set")

    client = CodeOcean(domain=domain, token=token)

    # Configure retry logic
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    client.session.mount(domain, adapter)

    # Initialize client components
    client.capsules = Capsules(client.session)
    client.computations = Computations(client.session)
    client.data_assets = DataAssets(client.session)

    return client


# def load_json_config(config_path):
#     """Load JSON configuration from a file"""
#     try:
#         with open(config_path, 'r') as f:
#             config = json.load(f)
  
#             # Create a list of settings from the config
#             settings_list = []
#             for setting in config['settings_list']:


#                 batch_settings = PipelineMonitorSettings.model_validate(
#                     {
#                         "run_params": {
#                             "capsule_id": setting['capsule_id'],
#                             "data_assets": [
#                                 DataAssetsRunParam(
#                                     id=asset['id'],
#                                     mount=asset['mount']
#                                 ) for asset in setting['assets_list']
#                             ],
#                             # "named_parameters": [
#                             #     NamedRunParam(param_name=key, value=value)
#                             #     for key, value in setting['named_parameters'].items()
#                             # ]
#                             "named_parameters": []
#                         },
#                         "capture_settings": CaptureSettings(
#                             tags=setting['tags'],
#                             process_name_suffix=setting['process_name_suffix']
#                         )
#                     }
#                 )
#                 settings_list.append(batch_settings)
#             return settings_list
#     except FileNotFoundError:
#         logging.error(f"Config file not found: {config_path}")
#         sys.exit(1)
#     except json.JSONDecodeError as e:
#         logging.error(f"Invalid JSON in config file: {e}")
#         sys.exit(1)
#     except ValueError as e:
#         logging.error(f"Invalid configuration: {e}")
#         sys.exit(1)


def load_json_config(config_path):
    """Load JSON configuration from a file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)



            batch_settings = PipelineMonitorSettings.model_validate(
                {
                    "run_params": {
                        "capsule_id": setting['capsule_id'],
                        "data_assets": [
                            DataAssetsRunParam(
                                id=asset['id'],
                                mount=asset['mount']
                            ) for asset in setting['assets_list']
                        ],
                        # "named_parameters": [
                        #     NamedRunParam(param_name=key, value=value)
                        #     for key, value in setting['named_parameters'].items()
                        # ]
                        "named_parameters": []
                    },
                    "capture_settings": CaptureSettings(
                        tags=setting['tags'],
                        process_name_suffix=setting['process_name_suffix']
                    )
                    }
                )
                settings_list.append(batch_settings)
            return settings_list
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Invalid configuration: {e}")
        sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run CodeOcean Pipeline Monitor jobs in parallel"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        help="Maximum number of concurrent processes (default: 2 * CPU count)",
        default=None
    )
    return parser.parse_args()


def run_single_job(settings_json: str):
    """Run a single monitor job for one data asset
    
    Parameters
    ----------
    settings_json : str
        JSON string containing the job settings
    """
    try:
        # Create new client for each process
        client = setup_codeocean_client()
        
        # Debug logging
        logging.info(f"Parsed settings: {settings.model_dump_json(indent=2)}")

        # Create and run job
        job = PipelineMonitorJob(job_settings=settings, client=client)
        job.run_job()
        
    except Exception as e:
        logging.error(f"Error in process: {str(e)}")
        logging.error(f"Failed settings JSON: {settings_json}")
        raise  # Re-raise the exception for the caller to handle


def run_jobs(config_path):
    """Run multiple jobs from settings JSON strings"""
    setup_logging()
    
    settings_list = load_json_config(config_path)
    for settings_json in settings_list:
        try:
            run_single_job(settings_json)
        except Exception as e:
            logging.error(f"Job failed: {str(e)}")
            continue


def main():
    """Entry point of the script"""
    try:
        args = parse_args()
        setup_logging()
        run_jobs(args.config)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()