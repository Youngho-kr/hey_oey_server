from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import json

import time

app = Flask(__name__)
CORS(app)

messages = []
message = []

def make_answer(data):
    time.sleep(3)
    message.append({"user": "OEY", "message": data['message']})
    # print(type(data))

def send_answer(message):
    if len(message) == 0:
        return json.loads('{"user": "NO_MESSAGE", "message": ""}'), 200
    else:
        print(message[0])
        return message[0], 200
        # return json.loads('{"user": "NO_MESSAGE", "message": ""}'), 200
    # return json.loads('{"user": "NO_MESSAGE", "message": ""}')

@app.route('/messages', methods=['GET', 'POST'])
def handle_messages():
    if request.method == 'POST':
        data = request.get_json()
        messages.append({'user': data['user'], 'message': data['message']})
        make_answer(data)
        return jsonify({'status': 'success'}), 200
    elif request.method == 'GET':
        _message = send_answer(message)
        message.clear()
        return send_answer(_message)

if __name__ == '__main__':
    app.run(debug=True)
