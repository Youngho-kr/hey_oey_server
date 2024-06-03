from .open_model import *

def process_message(message):
    response, label = generate_and_classify(message)

    return response
