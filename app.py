from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
import torch

app = Flask(__name__)

MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"  # 250MB

@lru_cache()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

@app.route("/moderate", methods=["POST"])
def moderate():
    tokenizer, model = load_model()
    
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)  # Shorter max_length
    
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]
    
    # This model has 3 classes: positive, negative, neutral
    # Use negative score as toxicity
    toxic_score = float(scores[1])  # index 1 is negative
    
    return jsonify({"toxic_score": toxic_score})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

