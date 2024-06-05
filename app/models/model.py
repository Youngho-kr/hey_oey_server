from .open_model import *

def process_message(message):
    response, intent = generate_and_classify(message)

    return response, intent
