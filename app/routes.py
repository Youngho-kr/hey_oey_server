from flask import request, jsonify, Blueprint
from .models.model import process_message

api = Blueprint('api', __name__)

messages = []
responses = []

@api.route('/messages', methods=['GET', 'POST'])
def handle_messages():
    print("Handle message")
    if request.method == 'POST':
        data = request.get_json()
        messages.append({'user': data['user'], 'message': data['message']})
        response, intent = process_message(data['message'])
        responses.append({"user": "OEY", "message": response})
        return jsonify({'status': 'success'}), 200
    elif request.method == 'GET':
        if responses:
            response = responses.pop(0)
            return jsonify(response), 200
        else:
            return jsonify({"user": "NO_MESSAGE", "message": ""}), 200
