from dotenv import load_dotenv
import os


load_dotenv()

UPLOAD_URL = os.getenv("UPLOAD_URL")
ML_SIGNATURE = os.getenv("ML_SIGNATURE")
