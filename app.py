from flask import Flask, jsonify
from chatbot import Chatbot

app = Flask(__name__)
chatbot = Chatbot("intents")


@app.route("/<message>", methods=['GET'])
def index(message: str):
    answer = chatbot.predict(message)
    return jsonify(answer)


if __name__ == "__main__":
    app.run(debug=True)
