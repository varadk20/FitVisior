from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM


app = Flask(__name__)

llm = OllamaLLM(
    model="mistral:7b-instruct",
    base_url="http://localhost:11434",
)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"reply": "Please ask a question."}), 400

    reply = llm.invoke(user_message)

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)