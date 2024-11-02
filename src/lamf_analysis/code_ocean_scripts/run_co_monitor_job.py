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


def setup_logging():
    """Configure logging with timestamp in filename"""
    log_file = f"run_co_monitor_job_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This maintains console output
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
            return (
                config['capsule_id'],
                config['tags'],
                config['process_name_suffix'],
                config['assets_list']
            )
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        sys.exit(1)
    except KeyError as e:
        logging.error(f"Missing required field in config: {e}")
        logging.error("Required fields: capsule_id, tags, process_name_suffix, assets_list")
        sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run CodeOcean Pipeline Monitor job with configuration from JSON file"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON configuration file"
    )
    return parser.parse_args()


def run_monitor_job(config_path):
    """Main function to run the monitor job"""

    capsule_id, tags, process_name_suffix, assets_list = load_json_config(config_path)

    client = setup_codeocean_client()
    
    data_asset_list = [DataAssetsRunParam(id=asset_dict['id'], 
                                          mount=asset_dict['mount']) for asset_dict in assets_list]

    # Configure settings
    settings = PipelineMonitorSettings(
        run_params=RunParams(
            capsule_id=capsule_id,
            data_assets=data_asset_list,
        ),
        capture_settings=CaptureSettings(
            tags=tags,
            process_name_suffix=process_name_suffix,
        ),
    )

    job = PipelineMonitorJob(job_settings=settings, client=client)
    job.run_job()

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