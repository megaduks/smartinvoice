import requests
from typing import Dict
import numpy as np
import cv2
import json
import logging

from classifiers import InvoiceTextClassifier
from image_processing import InvoiceOCR
from pika import ConnectionParameters, PlainCredentials, BlockingConnection
from settings import UPLOAD_URL, ML_SIGNATURE, DOWNLOAD_URL, LOG_DIRECTORY
from settings import RABBITMQ_PASSWORD, RABBITMQ_LOGIN, RABBITMQ_SERVER, RABBITMQ_QUEUE_NAME, INVOICE_NER_MODEL, \
    INVOICE_EAST_MODEL, RABBITMQ_PORT

LOGGER = logging.getLogger(__name__)


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
        LOGGER.error("Failed to send.", exc_info=True)

    return response


def start_connection(parameters: ConnectionParameters) -> (BlockingConnection, BlockingConnection):
    """
    Establishes connection with the queue, returns a BlockingConnection instance
    :param parameters RABBITMQ connection parameters.
    :return: BlockingConnection instance and it's channel for connection.

    """
    connection = BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


def get_queue_info(parameters: ConnectionParameters, queue="smart-invoice") -> object:
    """
    Checks if queue exists and returns it's info.
    :param parameters: RABBITMQ connection parameters.
    :param queue: Queue name.
    """
    _, channel = start_connection(parameters)
    info = (channel.queue_declare(queue=queue, passive=True))
    channel.close()
    return info


def get_image_from_token(token: str) -> np.ndarray:
    """
    Downloads image from B-MIND, requires unique token.
    :param token: Download token.
    :return n.ndarray: Download image in a numpy matrix, CV friendly.
    :return None: If failed to download.

    """

    url = DOWNLOAD_URL + token
    image = None

    try:
        LOGGER.info("Attempting to download image")
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        LOGGER.error("Failed to download image", exc_info=True)
    else:
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


class Consumer(object):
    """
    Class for receiving and processing tasks from queue.
    """

    def __init__(self, parameters: ConnectionParameters, nlp_model_path, ocr_model_path, queue):
        self.connection = BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.queue = queue
        self.nlp = InvoiceTextClassifier(model_path=nlp_model_path)
        self.ocr = InvoiceOCR(model_path=ocr_model_path.as_posix())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.channel.cancel()
        self.channel.close()
        self.connection.close()

    def start(self):
        self.channel.basic_consume(queue=self.queue, on_message_callback=self.callback, auto_ack=True)
        self.channel.start_consuming()

    def callback(self, channel, method_frame, properties, body):
        """
        Processes a RABBITMQ task from queue, passes it through OCR and Ner,
        output is formatted to fit the scheme and sent.
        """
        LOGGER.info(f"Processing message {body}")
        data = json.load(body.decode('utf-8'))
        LOGGER.info(f"Received message : {data}")
        image = get_image_from_token(data["download_token"])
        if image is not None:
            logging.info("[INFO] Processing image")
            _, OCR_text_output = self.ocr.process_image(image)

            LOGGER.info("[INFO] Processing OCR Output")
            ner_results = self.nlp.predict(OCR_text_output)

            send_json(payload=ner_results, job_id=data['job_id'], file_id=data['file_id'])



if __name__ == '__main__':
    logging.basicConfig(filemode='a', filename=f"{LOG_DIRECTORY}queue_tasks.log",
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    credentials = PlainCredentials(RABBITMQ_LOGIN, RABBITMQ_PASSWORD)
    connection_parameters = ConnectionParameters(host=RABBITMQ_SERVER, port=RABBITMQ_PORT, virtual_host="/",
                                                 credentials=credentials)

    with Consumer(parameters=connection_parameters, nlp_model_path=INVOICE_NER_MODEL, ocr_model_path=INVOICE_EAST_MODEL,
                  queue=RABBITMQ_QUEUE_NAME) as consumer:
        logging.info("Waiting for messages.")
        consumer.start()
