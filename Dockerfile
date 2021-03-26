FROM python:3.7.10

RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Installing Tesseract as well as the model for Polish
RUN apt-get -y install tesseract-ocr=4.0.0-2
RUN apt-get install tesseract-ocr-pol
# Dependancies for opencv
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get clean


COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN python -m spacy download pl_core_news_lg


WORKDIR /smartinvoice
COPY src /smartinvoice
# Copying the more sensitive data
COPY authfile /smartinvoice/.env

