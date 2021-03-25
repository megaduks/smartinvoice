import spacy
import plac
import json
import logging

from settings import MATCHERS
from spacy.language import Language
from matchers import remove_REGON_token
from typing import Dict, List, Optional, Union
from pathlib import Path
from itertools import chain
from settings import INVOICE_NER_MODEL
from utils import clean_ocr

# from ludwig.api import LudwigModel
# import pandas as pd

Language.factories['remove_REGON_token'] = remove_REGON_token


def get_entity(lst: List[Dict], key: str, default: Union[str, bool] = None) -> Optional:
    value = default
    for k, v in chain.from_iterable(d.items() for d in lst):
        if k == key:
            value = v
    return value


ner2json = {
    'INVOICE_NO': 'numer_faktury',
    'INVOICE_TYPE': 'typ_faktury',
    'PURCHASE_DATE': 'data_sprzedazy',
    'ISSUE_DATE': 'data_wystawienia',
    'NIP_SELLER': 'nip_sprzedawcy',
    'NAME_SELLER': 'nazwa_sprzedawcy',
    'NIP_BUYER': 'nip_nabywcy',
    'NAME_BUYER': 'nazwa_nabywcy',
    'NET_AMOUNT_TOTAL': 'razem_kwota_netto',
    'GROSS_AMOUNT_TOTAL': 'razem_kwota_brutto',
    'TAX_AMOUNT_TOTAL': 'razem_kwota_podatku_vat',
    'PAYMENT_FORM': 'forma_platnosci',
    'PAYMENT_DUE': 'termin_platnosci',
    'BANK_ACCOUNT_NO': 'numer_konta_sprzedawcy',
    'NAME_PRODUCT': 'nazwa',
    'MEASURE_PRODUCT': 'miara',
    'QTY_PRODUCT': 'ilosc',
    'UNIT_PRICE_PRODUCT': 'cena_jednostkowa',
    'NET_AMOUNT_PRODUCT': 'wartosc_netto',
    'TAX_RATE_PRODUCT': 'stawka_podatku',
    'TAX_AMOUNT_PRODUCT': 'kwota_podatku',
    'GROSS_AMOUNT_PRODUCT': 'wartosc_brutto',
}

json2ner = {
    v: k
    for k, v in ner2json.items()
}


# class InvoicePhotoClassifier:
#
#     def __init__(self, model_path: Path):
#         self.model = LudwigModel.load(model_path)
#
#     def predict(self, image_file: Path) -> bool:
#         """Verifies if the photo contains an image of an invoice"""
#         df = pd.DataFrame(
#             {
#                 'image_path': [image_file]
#             }
#         )
#         prediction = self.model.predict(data_df=df)
#
#         return prediction.class_predictions[0] == 'invoice'
#

class InvoiceTextClassifier:

    def __init__(self, model_path: str):
        self.nlp = spacy.load(model_path)
        self.nlp_clean = spacy.load('pl_core_news_lg')
        self.matchers = MATCHERS

    def predict(self, document: str) -> List:
        """Applies spaCy NER model to an OCR image"""

        clean_document = clean_ocr.clean_ocr(document, nlp=self.nlp_clean, matchers=self.matchers)

        entities = []

        def strip_ent_labels(label: str) -> str:
            """Strips all spaCy ent labels (xxxDate, xxxTime, xxxMoney) from the entity body"""
            tokens_to_strip = ['%', 'zl', 'zł', 'PLN', 'PL']

            clean_label = [
                token
                for token in label.split()
                if not token.startswith('xxx')
                   and token not in tokens_to_strip
            ]
            return ' '.join(clean_label)

        doc = self.nlp(clean_document)

        for ent in doc.ents:
            ent_value = strip_ent_labels(doc[ent.start:ent.end].text)
            entities.append({ent.label_: ent_value})

        answer = {
            'numer_faktury': get_entity(entities, 'INVOICE_NO'),
            'typ_faktury': get_entity(entities, 'INVOICE_TYPE'),
            'data_sprzedazy': get_entity(entities, 'PURCHASE_DATE'),
            'data_wystawienia': get_entity(entities, 'ISSUE_DATE'),
            'nip_sprzedawcy': get_entity(entities, 'NIP_SELLER'),
            'nazwa_sprzedawcy': None,
            'nip_nabywcy': get_entity(entities, 'NIP_BUYER'),
            'nazwa_nabywcy': None,
            'towary': [
                {
                    'lp': None,
                    'nazwa': get_entity(entities, 'PRODUCT_NAME'),
                    'miara': get_entity(entities, 'MEASURE_PRODUCT'),
                    'ilosc': get_entity(entities, 'QTY_PRODUCT'),
                    'cena_jednostkowa': get_entity(entities, 'UNIT_PRICE'),
                    'wartosc_netto': get_entity(entities, 'NET_AMOUNT_PRODUCT'),
                    'stawka_podatku': get_entity(entities, 'TAX_RATE_PRODUCT'),
                    'kwota_podatku': get_entity(entities, 'TAX_AMOUNT_PRODUCT'),
                    'wartosc_brutto': get_entity(entities, 'GROSS_AMOUNT_PRODUCT')
                }
            ],
            'razem_kwota_netto': get_entity(entities, 'NET_AMOUNT_TOTAL'),
            'razem_kwota_brutto': get_entity(entities, 'GROSS_AMOUNT_TOTAL'),
            'razem_kwota_podatku_vat': get_entity(entities, 'TAX_AMOUNT_TOTAL'),
            'przedplata_kwota_netto': None,
            'przedplata_kwota_brutto': None,
            'przedplata_kwota_podatku_vat': None,
            'rozliczenie_vat': [
                {
                    'stawka_podatku': get_entity(entities, 'TAX_RATE_PRODUCT'),
                    'wartosc_netto': get_entity(entities, 'NET_AMOUNT_TOTAL'),
                    'podatek': get_entity(entities, 'TAX_AMOUNT_TOTAL'),
                    'wartosc_brutto': get_entity(entities, 'GROSS_AMOUNT_TOTAL')
                }
            ],
            'czy_usluga': False,
            'forma_platnosci': get_entity(entities, 'PAYMENT_FORM'),
            'termin_platnosci': get_entity(entities, 'PAYMENT_DUE'),
            'numer_konta_sprzedawcy': get_entity(entities, 'BANK_ACCOUNT_NO'),
            'do_zaplaty': None,
            'zaplacono': None,
            'zaplacono_w_dniu': None,
            'razem_do_zwrotu': None,
            'faktura_zaliczkowa': {
                'data': None,
                'numer': None,
                'wartosc': None
            }
        }

        return answer


class InvoiceNERClassifier:

    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp

    def predict(self, document: str) -> List:
        """Applies spaCy NER model to an OCR image"""

        RESULT = []

        def strip_ent_labels(label: str) -> str:
            """Strips all spaCy ent labels (xxxDate, xxxTime, xxxMoney) from the entity body"""
            tokens_to_strip = ['%', 'zl', 'zł', 'PLN', 'PL']

            clean_label = [
                token
                for token in label.split()
                if not token.startswith('xxx')
                   and token not in tokens_to_strip
            ]
            return ' '.join(clean_label)

        doc = self.nlp(document)

        for ent in doc.ents:
            ent_value = strip_ent_labels(doc[ent.start:ent.end].text)
            RESULT.append({ent.label_: ent_value})

        return RESULT


@plac.annotations(
    input_dir=("Input directory", "option", "i", Path),
    output_dir=("Output directory", "option", "o", Path),
    model=("Directory with invoice entity recognizer model", "option", "m", Path)
)
def main(input_dir: Path, output_dir: Path, model: Path):
    """Helper function to find the entity value from the list of dictionaries with discovered entities"""

    FORMAT = "%(levelname)-5s %(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)
    logging.info('Extracting NERs from OCR files')

    def get_entity(lst: List[Dict], key: str, default: Union[str, bool] = None) -> Optional:
        value = default
        for k, v in chain.from_iterable(d.items() for d in lst):
            if k == key:
                value = v
        return value

    """Loads the model, set up the pipeline and train the entity recognizer."""
    logging.info('Loading language model')
    nlp = spacy.load(model)
    clf = InvoiceNERClassifier(nlp=nlp)

    input_files = input_dir.glob('*.txt')

    logging.info('Processing OCR files')
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            entities = clf.predict(f.readline())

            answer = {
                'numer_faktury': get_entity(entities, 'INVOICE_NO'),
                'typ_faktury': get_entity(entities, 'INVOICE_TYPE'),
                'data_sprzedazy': get_entity(entities, 'PURCHASE_DATE'),
                'data_wystawienia': get_entity(entities, 'ISSUE_DATE'),
                'nip_sprzedawcy': get_entity(entities, 'NIP_SELLER'),
                'nazwa_sprzedawcy': None,
                'nip_nabywcy': get_entity(entities, 'NIP_BUYER'),
                'nazwa_nabywcy': None,
                'towary': [
                    {
                        'lp': None,
                        'nazwa': get_entity(entities, 'PRODUCT_NAME'),
                        'miara': get_entity(entities, 'MEASURE_PRODUCT'),
                        'ilosc': get_entity(entities, 'QTY_PRODUCT'),
                        'cena_jednostkowa': get_entity(entities, 'UNIT_PRICE'),
                        'wartosc_netto': get_entity(entities, 'NET_AMOUNT_PRODUCT'),
                        'stawka_podatku': get_entity(entities, 'TAX_RATE_PRODUCT'),
                        'kwota_podatku': get_entity(entities, 'TAX_AMOUNT_PRODUCT'),
                        'wartosc_brutto': get_entity(entities, 'GROSS_AMOUNT_PRODUCT')
                    }
                ],
                'razem_kwota_netto': get_entity(entities, 'NET_AMOUNT_TOTAL'),
                'razem_kwota_brutto': get_entity(entities, 'GROSS_AMOUNT_TOTAL'),
                'razem_kwota_podatku_vat': get_entity(entities, 'TAX_AMOUNT_TOTAL'),
                'przedplata_kwota_netto': None,
                'przedplata_kwota_brutto': None,
                'przedplata_kwota_podatku_vat': None,
                'rozliczenie_vat': [
                    {
                        'stawka_podatku': get_entity(entities, 'TAX_RATE_PRODUCT'),
                        'wartosc_netto': get_entity(entities, 'NET_AMOUNT_TOTAL'),
                        'podatek': get_entity(entities, 'TAX_AMOUNT_TOTAL'),
                        'wartosc_brutto': get_entity(entities, 'GROSS_AMOUNT_TOTAL')
                    }
                ],
                'czy_usluga': False,
                'forma_platnosci': get_entity(entities, 'PAYMENT_FORM'),
                'termin_platnosci': get_entity(entities, 'PAYMENT_DUE'),
                'numer_konta_sprzedawcy': get_entity(entities, 'BANK_ACCOUNT_NO'),
                'do_zaplaty': None,
                'zaplacono': None,
                'zaplacono_w_dniu': None,
                'razem_do_zwrotu': None,
                'faktura_zaliczkowa': {
                    'data': None,
                    'numer': None,
                    'wartosc': None
                }
            }

            output_file = output_dir / f"{input_file.stem}.json"
            with open(output_file, 'w') as o:
                json.dump(answer, o)


if __name__ == '__main__':
    plac.call(main)
