from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

messages = []

def make_answer(data):
    messages.append({'user': 'OEY', 'message': f'My answer for ["{data['message']}]: GOOD!"'})

@app.route('/messages', methods=['GET', 'POST'])
def handle_messages():
    if request.method == 'POST':
        data = request.get_json()
        messages.append({'user': data['user'], 'message': data['message']})
        make_answer(data)
        return jsonify({'status': 'success'}), 200
    elif request.method == 'GET':
        return jsonify(messages), 200

if __name__ == '__main__':
    app.run(debug=True)
