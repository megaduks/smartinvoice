import json
import spacy
import cv2
import logging
import os

from settings import RABBITMQ_PASSWORD, RABBITMQ_LOGIN, RABBITMQ_SERVER, RABBITMQ_EXCHANGE_NAME, INVOICE_NER_MODEL, INVOICE_EAST_MODEL
from queue_tools import send_json, get_image_from_token, load_schema
from image_processing import InvoiceOCR
from classifiers import InvoiceNERClassifier
from typing import Dict, List
from pika import ConnectionParameters, PlainCredentials, BlockingConnection
from utils.reformat_data import reformat_json

if __name__ == '__main__':
    nlp = spacy.load(INVOICE_NER_MODEL)
    clf = InvoiceNERClassifier(nlp=nlp)
    OCR = InvoiceOCR(model_path=INVOICE_EAST_MODEL.as_posix())

    path_to_img = "21011f80ffffb199.png"

    image = cv2.imread(path_to_img)
    results, OCR_text = OCR.process_image(image)
    ner_results = clf.predict(OCR_text)

    base = os.path.basename(path_to_img)
    imgName, ext = base.split(".")
    with open(f"{imgName}.txt", "w") as file:
        file.write(OCR_text)

    with open('data.json', 'w') as outfile:
        json.dump(ner_results, outfile)

    ner_results = reformat_json(ner_results)

    with open('data_ref.json', 'w') as outfile:
        json.dump(ner_results, outfile)
