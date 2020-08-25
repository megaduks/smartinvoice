import requests
from typing import Dict
from settings import UPLOAD_URL, ML_SIGNATURE, DOWNLOAD_URL
import numpy as np
import cv2
import json


def sendJSON(data: Dict, job_id: str, file_id: str) -> object:
    # Takes in a dictionary, appends job and file id and  converts into json and sends to url.
    # Returns a request object with the return code and data.

    headers = {"X-ML-Signature": ML_SIGNATURE}
    data['job_id'] = job_id
    data['file_id'] = file_id

    try:
        response = requests.request(method="POST", url=UPLOAD_URL, json=data, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print("Failed to send")
        print(response.text)
    print(response)
    return response


def getImageFromToken(token: str) -> object:
    # Takes in the download token and requests the content from the B-MIND image host

    url = DOWNLOAD_URL + token

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print("Failed to download image:")
        print(err)
        return None
    
    image = np.asarray(bytearray(response.raw.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def generateTestJSON():
    with open("test.json") as f:
        data = json.load(f)
    return data

