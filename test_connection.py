from settings import RABBITMQ_PASSWORD, RABBITMQ_LOGIN, RABBITMQ_SERVER
import pika


credentials = pika.PlainCredentials(RABBITMQ_LOGIN, RABBITMQ_PASSWORD)

parameters = pika.ConnectionParameters(host=RABBITMQ_SERVER, port=5672, virtual_host="/", credentials=credentials)

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