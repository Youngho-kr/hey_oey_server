from .open_model import *
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, "close_model"))
print(os.path.join(current_dir, "close_model"))
from .close_model.application import *

# Close domain 모델검색 후 올바른 대답을 생성하지 못한 경우 Open domain 모델 응답 생성
def process_message(message):
    close_response = request_chat(message)
    print(close_response)
    close_state = close_response["state"]
    close_message = close_response["answer"]
    
    if close_state == "SUCCESS":
        return close_message, "경청"

    elif close_state == "FALLBACK":
        return generate_and_classify(message)

    else:
        return "자세히 말씀해주세요", "경청"

