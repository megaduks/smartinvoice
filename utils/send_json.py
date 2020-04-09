import requests
import os
from typing import Dict
from dotenv import load_dotenv


def send(data: Dict, job_id: str, file_id: str) -> object:
    # Takes in a dictionary, appends job and file id and  converts into json and sends to url.
    # Returns return code from the site.
    load_dotenv()
    url = os.getenv("UPLOAD_URL")
    signature = os.getenv("ML_SIGNATURE")
    headers = {"X-ML-Signature": signature}

    data['job_id'] = job_id
    data['file_id'] = file_id

    response = requests.request(method='POST', url=url, json=data, headers=headers)

    return response
