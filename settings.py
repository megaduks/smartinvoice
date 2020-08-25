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

INVOICE_IMAGE_MODEL = Path('experimental/ludwig/invoice_photo/results/experiment_run_10/model')
INVOICE_NER_MODEL = Path('experimental/prodigy/invoice_model_final')

MODELS = {
    'NIP': {
        'model_path': 'models/nip_model',
        'matcher_name': 'nip_matcher',
        'matcher_factory': NIPMatcher,
    },
    'BANK_ACCOUNT_NO': {
        'model_path': 'models/bank_account_model',
        'matcher_name': 'bank_account_matcher',
        'matcher_factory': BankAccountMatcher,
    },
    'REGON': {
        'model_path': 'models/regon_model',
        'matcher_name': 'regon_matcher',
        'matcher_factory': REGONMatcher,
    },
    'INVOICE_NUMBER': {
        'model_path': 'models/invoice_number_model',
        'matcher_name': 'invoice_number_matcher',
        'matcher_factory': InvoiceNumberMatcher,
    },
    'GROSS_VALUE': {
        'model_path': 'models/gross_value_model',
        'matcher_name': 'gross_value_matcher',
        'matcher_factory': GrossValueMatcher,
    },
    'DATE': {
        'model_path': 'models/date_model',
        'matcher_name': 'date_matcher',
        'matcher_factory': DateMatcher,
    },
    'MONEY': {
        'model_path': 'models/money_model',
        'matcher_name': 'money_matcher',
        'matcher_factory': MoneyMatcher,
    },
}
