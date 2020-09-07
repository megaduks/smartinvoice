import requests
from typing import Dict
from settings import UPLOAD_URL, ML_SIGNATURE, DOWNLOAD_URL
import numpy as np
import cv2
import json


def send_json(payload: Dict, job_id: str, file_id: str) -> requests.request:
    """
    Takes in a dictionary, appends job and file id and  converts into json and sends to url.

    :param payload : Data to be sent in python dictionary.
    :param job_id: Job ID to be appended to data, required for sending.
    :param file_id: File ID to be appended to data, required for sending.

    :return requests.response: Response from the B-MIND app, 200 upon successful delivery.

    """

    headers = {"X-ML-Signature": ML_SIGNATURE}
    payload['job_id'] = job_id
    payload['file_id'] = file_id
    try:
        response = requests.request(method="POST", url=UPLOAD_URL, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print("Failed to send")
        print(err)

    return response


def get_image_from_token(token: str) -> np.ndarray:
    """
    Downloads image from B-MIND, requires unique token.
    :param token: Download token.
    :return n.ndarray: Download image in a numpy matrix, CV friendly.
    :return None: If failed to download.

    """

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


def load_schema(filename: str) -> Dict:
    """
    Loads json from a file.
    """
    with open(filename) as f:
        data = json.load(f)
    return data
