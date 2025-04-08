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

# logging dir
LOG_DIR = Path("/home/matt.davis/code/mdev/src/mdev/co_pipeline_jobs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging():
    """Configure logging with timestamp in filename"""
    # get current script name
    script_name = os.path.basename(__file__)
    log_file = LOG_DIR / f"{script_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"

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
            
            # Create a list of settings, one for each batch
            settings_list = []
            for batch in config['assets_list']:
                # if subject_id is in the batch, pop out
                if "subject_id" in batch:
                    subject_id = batch.pop("subject_id")
                else:
                    subject_id = None

                print(subject_id)

                # custom metadata
                custom_metadata = config['custom_metadata'] if config['custom_metadata'] is not None else {}
                if subject_id is not None:
                    custom_metadata['subject id'] = subject_id

                # tags
                tags = (config['tags'].copy() if config['tags'] is not None else [])
                if subject_id is not None:
                    tags.append(subject_id)

                batch_settings = PipelineMonitorSettings.model_validate(
                    {
                        "run_params": {
                            "capsule_id": config['capsule_id'],
                            # "data_assets": [
                            #     DataAssetsRunParam(
                            #         id=asset['id'],
                            #         mount=asset['mount']
                            #     ) for asset in batch
                            # ]
                            "data_assets": [batch]
                        },
                        "capture_settings": CaptureSettings(
                            tags=tags,
                            process_name_suffix=config['process_name_suffix'],
                            custom_metadata=custom_metadata
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
    # dry-run flag
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the script"
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


def run_monitor_job(config_path, max_processes=None, dry_run=False):
    """Main function to run the monitor jobs in parallel"""
    # Load settings list (one per batch)
    settings_list = load_json_config(config_path)

    if max_processes is None:
        max_processes = multiprocessing.cpu_count() * 2

    logging.info(f"Running with maximum {max_processes} concurrent processes")

    # Create job settings JSON for each batch
    job_settings_list = [
        settings.model_dump_json() for settings in settings_list
    ]

    # Create process pool and submit jobs with delay
    if dry_run:
        logging.info("Dry run, not submitting jobs\n-")
        for settings_json in job_settings_list:
            logging.info(f"Dummy job: {settings_json}\n--------------------------------\n")
        return
    with Pool(processes=max_processes) as pool:
        results = []
        for settings_json in job_settings_list:
            if results:
                time.sleep(180)

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
        run_monitor_job(args.config, args.max_processes, args.dry_run)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()