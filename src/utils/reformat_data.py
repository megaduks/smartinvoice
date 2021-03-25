from typing import Dict, List, Optional, Union
from itertools import chain


def get_entity(lst: List[Dict], key: str, default: Union[str, bool] = None) -> Optional:
    value = default
    for k, v in chain.from_iterable(d.items() for d in lst):
        if k == key:
            value = v
    return value


def reformat_json(json_text: List[Dict]) -> Dict:

    answer = {
        'numer_faktury': get_entity(json_text, 'INVOICE_NO'),
        'typ_faktury': get_entity(json_text, 'INVOICE_TYPE'),
        'data_sprzedazy': get_entity(json_text, 'PURCHASE_DATE'),
        'data_wystawienia': get_entity(json_text, 'ISSUE_DATE'),
        'nip_sprzedawcy': get_entity(json_text, 'NIP_SELLER'),
        'nazwa_sprzedawcy': None,
        'nip_nabywcy': get_entity(json_text, 'NIP_BUYER'),
        'nazwa_nabywcy': None,
        'towary': [
            {
                'lp': None,
                'nazwa': get_entity(json_text, 'PRODUCT_NAME'),
                'miara': get_entity(json_text, 'MEASURE_PRODUCT'),
                'ilosc': get_entity(json_text, 'QTY_PRODUCT'),
                'cena_jednostkowa': get_entity(json_text, 'UNIT_PRICE'),
                'wartosc_netto': get_entity(json_text, 'NET_AMOUNT_PRODUCT'),
                'stawka_podatku': get_entity(json_text, 'TAX_RATE_PRODUCT'),
                'kwota_podatku': get_entity(json_text, 'TAX_AMOUNT_PRODUCT'),
                'wartosc_brutto': get_entity(json_text, 'GROSS_AMOUNT_PRODUCT')
            }
        ],
        'razem_kwota_netto': get_entity(json_text, 'NET_AMOUNT_TOTAL'),
        'razem_kwota_brutto': get_entity(json_text, 'GROSS_AMOUNT_TOTAL'),
        'razem_kwota_podatku_vat': get_entity(json_text, 'TAX_AMOUNT_TOTAL'),
        'przedplata_kwota_netto': None,
        'przedplata_kwota_brutto': None,
        'przedplata_kwota_podatku_vat': None,
        'rozliczenie_vat': [
            {
                'stawka_podatku': get_entity(json_text, 'TAX_RATE_PRODUCT'),
                'wartosc_netto': get_entity(json_text, 'NET_AMOUNT_TOTAL'),
                'podatek': get_entity(json_text, 'TAX_AMOUNT_TOTAL'),
                'wartosc_brutto': get_entity(json_text, 'GROSS_AMOUNT_TOTAL')
            }
        ],
        'czy_usluga': False,
        'forma_platnosci': get_entity(json_text, 'PAYMENT_FORM'),
        'termin_platnosci': get_entity(json_text, 'PAYMENT_DUE'),
        'numer_konta_sprzedawcy': get_entity(json_text, 'BANK_ACCOUNT_NO'),
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

