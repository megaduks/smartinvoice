from settings import RABBITMQ_PASSWORD, RABBITMQ_LOGIN, RABBITMQ_SERVER, RABBITMQ_EXCHANGE_NAME
import requests
import pika
import typing
import json
import logging
from net_tools import sendJSON, getImageFromToken
import cv2

credentials = pika.PlainCredentials(RABBITMQ_LOGIN, RABBITMQ_PASSWORD)
parameters = pika.ConnectionParameters(host=RABBITMQ_SERVER, port=5672, virtual_host="/", credentials=credentials)
exchange_name = RABBITMQ_EXCHANGE_NAME


def processResponse(response, method_frame):
    data = json.loads(response.decode('utf-8'))
    print(f"Received message : {data}")
    print("Downloading image")
    image = getImageFromToken(data["download_token"])

    # save image if download succeeds
    if image is not None:

        filename = f"{method_frame.delivery_tag}.jpg"
        cv2.imwrite(filename, image)



    #payload = generateTestJSON()

    #print("Sending JSON")
    #sendJSON(data=payload, job_id=data['job_id'], file_id=data['file_id'])


def testConnection(parameters):
    # try to establish connection and check its status
    try:
        connection = pika.BlockingConnection(parameters)
        if connection.is_open:
            print('OK')
            connection.close()
            exit(0)
    except Exception as error:
        print('Error:', error.__class__.__name__)
        exit(1)


def startConnection(parameters):
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    return connection, channel


def callback(channel, method_frame, properties, body):
    print(method_frame)
    print(properties)
    print(body)

    processResponse(body)

    channel.basic_ack(method_frame.delivery_tag)


def getQueueInfo(parameters):
    _, channel = startConnection(parameters)
    print(channel.queue_declare(queue="smart-invoice", passive=True))
    channel.close()


def startConsuming(parameters):

    connection, channel = startConnection(parameters)
    for method_frame, properties, body in channel.consume("smart-invoice"):

        print(method_frame)
        print(properties)
        print(body)

        processResponse(body, method_frame)

        channel.basic_ack(method_frame.delivery_tag)

        if method_frame.delivery_tag == 1:
            break

    requeued_messages = channel.cancel()
    print('Requeued %i messages' % requeued_messages)

    channel.close()
    connection.close()

getQueueInfo(parameters)