from dotenv import load_dotenv
import os

from matchers import *
from pathlib import Path

load_dotenv()

#OCR Parameters
OCR_MIN_CONFIDENCE = 0.5
OCR_PADDING = 0.1
OCR_TESSERACT_CONFIG = "-l pol --oem 1  --psm 7"

UPLOAD_URL = os.getenv("UPLOAD_URL")
ML_SIGNATURE = os.getenv("ML_SIGNATURE")
DOWNLOAD_URL = os.getenv("DOWNLOAD_URL")

RABBITMQ_LOGIN = os.getenv("RABBITMQ_LOGIN")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_SERVER = os.getenv("RABBITMQ_SERVER")
RABBITMQ_EXCHANGE_NAME = os.getenv("EXCHANGE_NAME")

INVOICE_EAST_MODEL = Path('/models/frozen_east_text_detection.pb')
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
