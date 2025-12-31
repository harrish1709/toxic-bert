import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify
from profanity_check import predict_prob

app = Flask(__name__)

@app.route('/moderate', methods=['POST'])
def moderate():
    try:
        data = request.json
        text = data.get("text", "")
        
        if not text:
            return jsonify({"toxic_score": 0.0})

        score = predict_prob([text])[0]
        
        return jsonify({"toxic_score": float(score)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
