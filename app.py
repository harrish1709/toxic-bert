from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# --- Load Toxic-BERT locally ---
MODEL_NAME = "unitary/toxic-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

label_to_index = {"toxic": 0}   # toxic-bert label ordering


@app.route("/moderate", methods=["POST"])
def moderate():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]

        # Extract toxic score (index 0)
        toxic_score = float(scores[label_to_index["toxic"]])

        return jsonify({"toxic_score": toxic_score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
