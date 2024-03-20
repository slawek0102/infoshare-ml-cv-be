from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test")
def test():
    return jsonify({"message": "This is test"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)