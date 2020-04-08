import requests
from typing import Dict


def send(data: Dict, job_id: str, file_id: str) -> object:
    # Takes in a dictionary, appends job and file id and  converts into json and sends to url.
    # Returns return code from the site.

    url = "https://api-smart-invoice-dev.b-mind.pl/api/v1/webhook/ml/send"
    headers = {"X-ML-Signature": "QpachX9AAhKD8CB3USMk5nrMee8QvdLvYLGdhFdDpAr2vyzkK5t3q3T9wbxstMXG"}

    data['job_id'] = job_id
    data['file_id'] = file_id

    response = requests.request(method='POST', url=url, json=data, headers=headers)

    return response
