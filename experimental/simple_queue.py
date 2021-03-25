import json
import cv2

from settings import RABBITMQ_PASSWORD, RABBITMQ_LOGIN, RABBITMQ_SERVER, RABBITMQ_EXCHANGE_NAME
from queue_tools import send_json, get_image_from_token, load_schema

from typing import Dict, List
from pika import ConnectionParameters, PlainCredentials, BlockingConnection

credentials = PlainCredentials(RABBITMQ_LOGIN, RABBITMQ_PASSWORD)
connection_parameters = ConnectionParameters(host=RABBITMQ_SERVER, port=5672, virtual_host="/", credentials=credentials)
exchange_name = RABBITMQ_EXCHANGE_NAME


def processResponse(response, method_frame):

    data = json.loads(response.decode('utf-8'))
    print(f"Received message : {data}")
    print("[INFO] Downloading image.")
    image = get_image_from_token(data["download_token"])

    # save image if download succeeds
    if image is not None:
        filename = f"{method_frame.delivery_tag}.png"
        cv2.imwrite(filename, image)

    payload = load_schema("static.json")

    print("[INFO] Sending JSON")
    send_json(payload=payload, job_id=data['job_id'], file_id=data['file_id'])


def update_payload(payload: Dict, dict_output: Dict) -> Dict:

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
    for key in entities_to_remove:
        payload.pop(key, None)

    return payload


def test_connection(parameters: ConnectionParameters):
    # try to establish connection and check its status
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
    connection = BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


def callback(channel, method_frame, properties, body, OCR):
    pass


def get_queue_info(parameters: ConnectionParameters):
    _, channel = start_connection(parameters)
    print(channel.queue_declare(queue="smart-invoice", passive=True))
    channel.close()


def start_consuming(parameters: ConnectionParameters):

    connection, channel = start_connection(parameters)
    for method_frame, properties, body in channel.consume("smart-invoice"):

        processResponse(body, method_frame)
        channel.basic_ack(method_frame.delivery_tag)

        if method_frame.delivery_tag == 0:
            break

    requeued_messages = channel.cancel()
    print('Requeued %i messages' % requeued_messages)

    channel.close()
    connection.close()


if __name__ == '__main__':
    get_queue_info(connection_parameters)
    start_consuming(connection_parameters)
