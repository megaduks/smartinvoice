import pika
import json
import spacy
import cv2

from settings import RABBITMQ_PASSWORD, RABBITMQ_LOGIN, RABBITMQ_SERVER, RABBITMQ_EXCHANGE_NAME
from net_tools import sendJSON, getImageFromToken
from image_processing import InvoiceOCR
from classifiers import InvoiceNERClassifier
from typing import Dict, List

credentials = pika.PlainCredentials(RABBITMQ_LOGIN, RABBITMQ_PASSWORD)
parameters = pika.ConnectionParameters(host=RABBITMQ_SERVER, port=5672, virtual_host="/", credentials=credentials)
exchange_name = RABBITMQ_EXCHANGE_NAME
model = "/home/oliver/Documents/smartinvoice/experimental/prodigy/invoice_model_final/"


def processResponse(response, method_frame, OCR, NER):
    data = json.loads(response.decode('utf-8'))
    print(f"Received message : {data}")
    print("[INFO] Downloading image. ")
    image = getImageFromToken(data["download_token"])
    ner_results = []

    # save image if download succeeds
    if image is not None:
        filename = f"{method_frame.delivery_tag}.jpg"
        cv2.imwrite(filename, image)

        print("[INFO] Processing image")
        OCR_text_output = OCR.process_image(image)

        print("[INFO] Processing OCR Output")
        ner_results = NER.predict(OCR_text_output)

    payload = {}

    for d in ner_results:
        payload.update(d)

    print(payload)
    payload = update_payload(payload)
    print(payload)

    print("[INFO] Sending JSON")
    sendJSON(payload=payload, job_id=data['job_id'], file_id=data['file_id'])


def update_payload(payload: Dict) -> Dict:

    payload["rozliczenie_vat"].append({"stawka_podatku": 0,
                                       "wartosc_netto": 0,
                                       "podatek": 0,
                                       "wartosc_brutto": 0})

    payload["towary"] = [{"lp": 0,
                          "nazwa": "",
                          "miara": "",
                          "ilosc": "",
                          "cena_jednostkowa": 0,
                          "wartosc_netto": 0,
                          "stawka_podatku": 0,
                          "kwota_podatku": 0,
                          "wartosc_brutto": 0}]

    keys_tax = payload["towary"][0].keys() + payload["rozliczenie_vat"][0].keys()
    keys_product = payload["towary"][0].keys()
    keys_to_remove = keys_tax + keys_product

    for key in payload.keys():
        if key in keys_tax:
            payload["rozliczenie_vat"][0][key] = payload[key]

    for key in payload.keys():
        if key in keys_product:
            payload["rozliczenie_vat"][0][key] = payload[key]

    if "numer_faktury" not in payload.keys():
        payload["numer_faktury"] = ""
    if "nip_sprzedawcy" not in payload.keys():
        payload["nip_sprzedawcy"] = ""
    if "typ_faktury" not in payload.keys():
        payload["typ_faktury"] = "faktura_vat"
    if "razem_kwota_brutto" not in payload.keys():
        payload["razem_kwota_brutto"] = 0

    payload = clean_payload(payload, keys_to_remove)

    return payload


def clean_payload(payload: Dict, entities_to_remove: List) -> Dict:
    for key in entities_to_remove:
        payload.pop(key, None)

    return payload


def test_connection(parameters):
    # try to establish connection and check its status
    try:
        connection = pika.BlockingConnection(parameters)
        if connection.is_open:
            print('OK')
            connection.close()
            exit(0)
    except Exception as error:
        print('Error:', error.__class__.__name__)
        exit(1)


def start_connection(parameters):
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


def callback(channel, method_frame, properties, body, OCR):
    pass


def get_queue_info(parameters):
    _, channel = start_connection(parameters)
    print(channel.queue_declare(queue="smart-invoice", passive=True))
    channel.close()


def start_consuming(parameters, OCR, NLP):
    connection, channel = start_connection(parameters)
    for method_frame, properties, body in channel.consume("smart-invoice"):

        processResponse(body, method_frame, OCR, NLP)
        channel.basic_ack(method_frame.delivery_tag)

        if method_frame.delivery_tag == 1:
            break

    requeued_messages = channel.cancel()
    print('Requeued %i messages' % requeued_messages)

    channel.close()
    connection.close()


if __name__ == '__main__':
    get_queue_info(parameters)
    nlp = spacy.load(model)
    clf = InvoiceNERClassifier(nlp=nlp)
    OCR = InvoiceOCR(model_path="/home/oliver/Documents/smartinvoice/models/frozen_east_text_detection.pb")
    start_consuming(parameters, OCR, clf)
