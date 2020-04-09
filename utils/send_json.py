import requests
from typing import Dict
from settings import UPLOAD_URL, ML_SIGNATURE


def send(data: Dict, job_id: str, file_id: str) -> object:
    # Takes in a dictionary, appends job and file id and  converts into json and sends to url.
    # Returns a request object with the return code and data.

    headers = {"X-ML-Signature": ML_SIGNATURE}
    data['job_id'] = job_id
    data['file_id'] = file_id

    response = requests.request(method='POST', url=UPLOAD_URL, json=data, headers=headers)

    return response
