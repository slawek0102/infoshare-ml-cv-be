from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test")
def test():
    return jsonify({"message": "This is test"})