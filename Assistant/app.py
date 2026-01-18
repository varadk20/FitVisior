from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM


app = Flask(__name__)

llm = OllamaLLM(
    model="mistral:7b-instruct",
    base_url="http://localhost:11434",
)
