from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

MODEL_NAME = "unitary/toxic-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load Toxic-BERT in FP16 (cuts memory by 50%)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).eval()

label_to_index = {"toxic": 0}


@app.route("/moderate", methods=["POST"])
def moderate():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        inputs = tokenizer(text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0]

        toxic_score = float(scores[label_to_index["toxic"]])

        return jsonify({"toxic_score": toxic_score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
