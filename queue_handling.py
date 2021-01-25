import json
import spacy
import cv2
import logging

from settings import RABBITMQ_PASSWORD, RABBITMQ_LOGIN, RABBITMQ_SERVER, RABBITMQ_EXCHANGE_NAME, INVOICE_NER_MODEL, INVOICE_EAST_MODEL
from net_tools import send_json, get_image_from_token, load_schema
from image_processing import InvoiceOCR
from classifiers import InvoiceNERClassifier
from typing import Dict, List
from pika import ConnectionParameters, PlainCredentials, BlockingConnection

credentials = PlainCredentials(RABBITMQ_LOGIN, RABBITMQ_PASSWORD)
connection_parameters = ConnectionParameters(host=RABBITMQ_SERVER, port=5672, virtual_host="/", credentials=credentials)
exchange_name = RABBITMQ_EXCHANGE_NAME
logging.basicConfig(filemode='a', filename="queue_tasks.log", format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def process_message(rabbit_task, method_frame, OCR: InvoiceOCR, NER: InvoiceNERClassifier):

    """
    Processes a RABBITMQ task from queue, processes it through OCR and Ner,
    output is formatted to fit the scheme and sent, image is saved.
    :param rabbit_task: 
    :param method_frame: AMQP frame with RPC response.
    :param OCR: Text detector and ocr.
    :param NER: Ner Classifier

    """

    data = json.loads(rabbit_task.decode('utf-8'))
    logging.info(f"Received message : {data}")
    logging.debug("Downloading image.")
    image = get_image_from_token(data["download_token"])
    ner_results = []

    # save image if download succeeds
    if image is not None:
        filename = f"{data['file_id']}.png"
        cv2.imwrite(filename, image)

        logging.info("[INFO] Processing image")
        OCR_text_output, _ = OCR.process_image(image)

        logging.info("[INFO] Processing OCR Output")
        ner_results = NER.predict(OCR_text_output)

    payload = load_schema("default.json")
    NER_output = {}

    for d in ner_results:
        NER_output.update(d)

    logging.debug(NER_output)
    payload = update_payload(payload, NER_output)
    logging.debug(payload)

    logging.info("Sending JSON")
    send_json(payload=payload, job_id=data['job_id'], file_id=data['file_id'])


def update_payload(payload: Dict, dict_output: Dict) -> Dict:
    """
    Updates payload with output from NER, formatted to fit the v0.5 schema
    :return Updated payload.
    """

    keys_tax = ['stawka_podatku', 'wartosc_netto', 'podatek', 'wartosc_brutto']
    keys_product = ['lp', 'nazwa', 'miara', 'ilosc', 'cena_jednostkowa', 'wartosc_netto', 'stawka_podatku',
                    'kwota_podatku', 'wartosc_brutto']

    for key in dict_output.copy().keys():
        if key in keys_tax:
            payload["rozliczenie_vat"][0][key] = dict_output.pop(key, None)

        elif key in keys_product:
            payload["towary"][0][key] = dict_output.pop(key, None)

    payload.update(dict_output)

    return payload


def clean_payload(payload: Dict, entities_to_remove: List) -> Dict:
    """
    Removes redundant data.
    """
    for key in entities_to_remove:
        payload.pop(key, None)

    return payload


def test_connection(parameters: ConnectionParameters):
    """
    Tests connection with server, exits if none is established.
    :param parameters: RABBITMQ connection parameters.
    """
    try:
        connection = BlockingConnection(parameters)
        if connection.is_open:
            print('OK')
            connection.close()
            exit(0)
    except Exception as error:
        print('Error:', error.__class__.__name__)
        exit(1)


def start_connection(parameters: ConnectionParameters) -> (BlockingConnection, BlockingConnection):
    """
    Establishes connection with the queue, returns a BlockingConnection instance
    :param parameters RABBITMQ connection parameters.
    :return: BlockingConnection instance and it's channel for connection.

    """
    connection = BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


def callback(channel, method_frame, properties, body):
    pass


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


def start_consuming(parameters: ConnectionParameters, OCR: InvoiceOCR, NLP: InvoiceNERClassifier, message_limit=1):
    """
    Processes messages from queue, limited in amount of tasks processed.

    :param parameters: RABBITMQ connection parameters.
    :param OCR: OCR model for image processing.
    :param NLP: NLP model for NER.
    :param message_limit: Maximum number of processed messages.
    """
    connection, channel = start_connection(parameters)
    for method_frame, properties, body in channel.consume("smart-invoice"):

        process_message(body, method_frame, OCR, NLP)
        channel.basic_ack(method_frame.delivery_tag)

        if method_frame.delivery_tag == message_limit:
            break

    requeued_messages = channel.cancel()
    print('Requeued %i messages' % requeued_messages)

    channel.close()
    connection.close()


if __name__ == '__main__':
    logging.info(get_queue_info(connection_parameters))
    nlp = spacy.load(INVOICE_NER_MODEL)
    clf = InvoiceNERClassifier(nlp=nlp)
    OCR = InvoiceOCR(model_path=INVOICE_EAST_MODEL.as_posix())
    start_consuming(connection_parameters, OCR, clf)
