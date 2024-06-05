from .open_model import *
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, "close_model"))
print(os.path.join(current_dir, "close_model"))
from .close_model.application import *

def process_message(message):
    response, intent = generate_and_classify(message)

    return response, intent
