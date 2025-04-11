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
from pathlib import Path


def setup_logging(log_dir):
    """Configure logging with timestamp in filename"""
    if log_dir is None:
        print("No log directory provided, no logging will be done")
        return None
    
    try:
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        
        script_name = os.path.basename(__file__)
        log_file = log_dir / f"{script_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        # Remove any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create file handler with explicit encoding
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        
        logging.info(f"Logging to {log_file}")
        return log_file
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return None


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
            if 'capture_settings' in config.keys():
                if config['capture_settings'] is not None:
                    capture_settings = config['capture_settings']
                    assert len(capture_settings) == len(config['assets_list']), \
                        "Number of capture settings must match number of assets in assets_list"
            custom_metadata = {}
            if 'custom_metadata' in config.keys():
                custom_metadata = config['custom_metadata']
            if 'named_parameters' in config.keys():
                named_param_dict = config['named_parameters']
                named_param = []
                for key, value in named_param_dict.items():
                    named_param.append(NamedRunParam(param_name=key, value=f"{value}"))
            else:
                named_param = None
            

            settings_list = []
            for batch, capture_setting in zip(config['assets_list'], capture_settings):
                # tags
                tags = (config['tags'].copy() if config['tags'] is not None else [])
                
                # capture settings
                result_name = capture_setting['result_name'] if 'result_name' in capture_setting.keys() else None
                result_description = capture_setting['result_description'] if 'result_description' in capture_setting.keys() else None
                temp_capture_settings_dict = {
                    "process_name_suffix": config['process_name_suffix'] if 'process_name_suffix' in config.keys() else None,
                    "tags": tags,
                    "custom_metadata": custom_metadata,
                    "name": result_name,
                    "mount": result_name if result_name is not None else None,
                    "description": result_description,
                }
                capture_settings_dict = {k: v for k, v in temp_capture_settings_dict.items() if v is not None}
                current_capture_settings = CaptureSettings(
                    **capture_settings_dict
                )

                # job settings
                run_params = {
                    "capsule_id": config['capsule_id'],
                    "data_assets": batch,
                    "named_parameters": named_param,
                }
                run_params = {k: v for k, v in run_params.items() if v is not None}
                batch_settings = PipelineMonitorSettings.model_validate(
                    {
                        "run_params": run_params,
                        "alert_url": config['alert_url'] if 'alert_url' in config.keys() else None,
                        "capture_settings": current_capture_settings
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
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory to save log files (default: current directory)",
        default=None
    )
    # dry-run flag
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the script"
    )
    return parser.parse_args()


def run_single_job(settings_dict, log_dir=None):
    """Run a single monitor job for one data asset"""
    try:
        # Create new client for each process
        client = setup_codeocean_client()

        # Deserialize settings
        settings = PipelineMonitorSettings.model_validate_json(settings_dict)

        # Set up logging for this worker process
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            # Use the main log file but append process ID to distinguish worker logs
            script_name = os.path.basename(__file__)
            log_file = log_dir / f"{script_name}_{time.strftime('%Y%m%d_%H%M%S')}_worker{os.getpid()}.log"
            
            # Configure logging to append to the main log file
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - PID%(process)d - %(levelname)s - %(message)s'))
            logging.root.addHandler(file_handler)
            logging.root.setLevel(logging.INFO)

        asset_mount = settings.run_params.data_assets[0].mount
        logging.info(f"Starting job for asset {asset_mount}")
        
        job = PipelineMonitorJob(job_settings=settings, client=client)
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

    # Get the current log directory from the root logger's handlers
    log_dir = None
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            log_dir = str(Path(handler.baseFilename).parent)
            break

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
            # Pass log_dir to the worker process
            result = pool.apply_async(run_single_job, (settings_json, log_dir))
            results.append(result)

        for result in results:
            result.get()

    logging.info("All processes completed")


def main():
    """Entry point of the script"""
    try:
        args = parse_args()
        setup_logging(args.log_dir)
        run_monitor_job(args.config, args.max_processes, args.dry_run)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()