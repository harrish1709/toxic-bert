from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import torch

app = Flask(__name__)

MODEL_NAME = "martin-ha/toxic-comment-model"  # smaller model

@lru_cache()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True
    ).eval()
    return tokenizer, model


@app.route("/moderate", methods=["POST"])
def moderate():
    tokenizer, model = load_model()

    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.inference_mode():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]

    toxic_score = float(scores[1])  # index depends on model
    return jsonify({"toxic_score": toxic_score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
