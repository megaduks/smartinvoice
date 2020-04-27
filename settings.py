from dotenv import load_dotenv
import os

from matchers import NIPMatcher, BankAccountMatcher

load_dotenv()

UPLOAD_URL = os.getenv("UPLOAD_URL")
ML_SIGNATURE = os.getenv("ML_SIGNATURE")

RABBITMQ_LOGIN = os.getenv("RABBITMQ_LOGIN")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
RABBITMQ_SERVER = os.getenv("RABBITMQ_SERVER")

NER_MATCHERS = {
    'nip_matcher': NIPMatcher,
    'bank_account_matcher': BankAccountMatcher
}