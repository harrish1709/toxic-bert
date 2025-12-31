
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
moderator = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None
)

@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.json
    text = data.get("text", "")

    try:
        results = moderator(text)[0]
        toxic_score = next(
            (r["score"] for r in results if r["label"] == "toxic"),
            0
        )
        return jsonify({"toxic_score": toxic_score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
