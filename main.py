from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/echo', methods=['POST'])
def echo():
    data = request.json  # JSON 데이터를 파싱합니다.
    return jsonify({
        "message": data['message']  # 받은 메시지를 그대로 반환
    })

if __name__ == '__main__':
    app.run(debug=True)
