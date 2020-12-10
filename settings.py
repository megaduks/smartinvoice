from dotenv import load_dotenv
import os

from matchers import *
from pathlib import Path

load_dotenv()

UPLOAD_URL = os.getenv("UPLOAD_URL")
ML_SIGNATURE = os.getenv("ML_SIGNATURE")

RABBITMQ_LOGIN = os.getenv("RABBITMQ_LOGIN")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_SERVER = os.getenv("RABBITMQ_SERVER")

INVOICE_IMAGE_MODEL = Path('experimental/ludwig/invoice_photo/results/experiment_run_10/model')
INVOICE_NER_MODEL = Path('models/invoice_final_ner_model')

MODELS = {
    'NIP': {
        'matcher_name': 'nip_matcher',
        'matcher_factory': NIPMatcher,
    },
    'BANK_ACCOUNT_NO': {
        'matcher_name': 'bank_account_matcher',
        'matcher_factory': BankAccountMatcher,
    },
    'REGON': {
        'matcher_name': 'regon_matcher',
        'matcher_factory': REGONMatcher,
    },
    'INVOICE_NUMBER': {
        'matcher_name': 'invoice_number_matcher',
        'matcher_factory': InvoiceNumberMatcher,
    },
    'GROSS_VALUE': {
        'matcher_name': 'gross_value_matcher',
        'matcher_factory': GrossValueMatcher,
    },
    'DATE': {
        'matcher_name': 'date_matcher',
        'matcher_factory': DateMatcher,
    },
    'MONEY': {
        'matcher_name': 'money_matcher',
        'matcher_factory': MoneyMatcher,
    },
}
