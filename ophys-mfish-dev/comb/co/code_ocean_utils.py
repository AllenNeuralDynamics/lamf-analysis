
import os
import yaml
import requests
from typing import Optional, Union
from pathlib import Path



def attach_assets(data: Union[list, str, Path], api_secret_name = "CODEOCEAN_TOKEN"):
    """
    
    "API_SECRET"
    """

    if isinstance(data, (str, Path)):
        with open(data, mode="r") as f:
            data = yaml.safe_load(f)
    elif isinstance(data, list):
        for d in data:
            assert isinstance(d, dict), "All elements of the list must be dictionaries"

    url = f'https://codeocean.allenneuraldynamics.org/api/v1/capsules/{os.getenv("CO_CAPSULE_ID")}/data_assets'
    headers = {"Content-Type": "application/json"} 
    auth = ("'" + os.getenv(api_secret_name), "'")
    print(data)
    response = requests.post(url=url, headers=headers, auth=auth, json=data)
    print(response.text)