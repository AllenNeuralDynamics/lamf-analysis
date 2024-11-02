from aind_codeocean_pipeline_monitor.job import PipelineMonitorJob
from aind_codeocean_pipeline_monitor.models import (
    CaptureSettings,
    PipelineMonitorSettings,
)
from codeocean.capsule import Capsules
from codeocean.data_asset import DataAssets
from codeocean.computation import Computations, DataAssetsRunParam, RunParams
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


def setup_logging():
    """Configure logging with timestamp in filename"""
    log_file = f"run_co_monitor_job_parallel_{time.strftime('%Y%m%d_%H%M%S')}.log"

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


def load_json_config(config_path):
    """Load JSON configuration from a file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
            # Create settings object directly from JSON
            settings = PipelineMonitorSettings.model_validate(
                {
                    "run_params": {
                        "capsule_id": config['capsule_id'],
                        "data_assets": [
                            {
                                "id": asset['id'],
                                "mount": asset['mount']
                            } for asset in config['assets_list']
                        ]
                    },
                    "capture_settings": {
                        "tags": config['tags'],
                        "process_name_suffix": config['process_name_suffix']
                    }
                }
            )
            return settings
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


def run_single_job(settings_dict):
    """Run a single monitor job for one data asset"""
    try:
        # Create new client for each process
        client = setup_codeocean_client()
        
        # Deserialize settings
        settings = PipelineMonitorSettings.model_validate_json(settings_dict)
        
        job = PipelineMonitorJob(job_settings=settings, client=client)
        asset_mount = settings.run_params.data_assets[0].mount
        logging.info(f"Starting job for asset {asset_mount}")
        job.run_job()
        logging.info(f"Completed job for asset {asset_mount}")
    except Exception as e:
        logging.error(f"Error in process for asset {settings.run_params.data_assets[0].mount}: {e}")


def run_monitor_job(config_path, max_processes=None):
    """Main function to run the monitor jobs in parallel"""
    # Load full settings
    settings = load_json_config(config_path)
    
    if max_processes is None:
        max_processes = multiprocessing.cpu_count() * 2

    logging.info(f"Running with maximum {max_processes} concurrent processes")

    # Create individual settings for each asset
    job_settings_list = []
    for asset in settings.run_params.data_assets:
        single_asset_settings = PipelineMonitorSettings(
            run_params=RunParams(
                capsule_id=settings.run_params.capsule_id,
                data_assets=[asset]
            ),
            capture_settings=settings.capture_settings
        )
        job_settings_list.append(single_asset_settings.model_dump_json())

    # Create process pool and submit jobs with delay
    with Pool(processes=max_processes) as pool:
        results = []
        for settings_json in job_settings_list:
            if results:
                logging.info("Waiting 30 seconds before starting next job...")
                time.sleep(30)

            logging.info(f"Submitting new job")
            result = pool.apply_async(run_single_job, (settings_json,))
            results.append(result)

        for result in results:
            result.get()

    logging.info("All processes completed")


def main():
    """Entry point of the script"""
    try:
        args = parse_args()
        setup_logging()
        run_monitor_job(args.config)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()