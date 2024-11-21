from app import app, chatbot
from flask import request, jsonify, render_template

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = chatbot.get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
