from dotenv import load_dotenv
import os

from matchers import *

load_dotenv()

UPLOAD_URL = os.getenv("UPLOAD_URL")
ML_SIGNATURE = os.getenv("ML_SIGNATURE")

RABBITMQ_LOGIN = os.getenv("RABBITMQ_LOGIN")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_SERVER = os.getenv("RABBITMQ_SERVER")

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
    }
}

NOISE_CHARACTERS = '|{}[]()"!<>=^`~'