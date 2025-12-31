from flask import Flask, request, jsonify
from profanity_check import predict_prob

app = Flask(__name__)

@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.json
    text = data.get("text", "")
    
    # predict_prob returns a numpy array like [0.04], we take the first item
    # This runs in milliseconds and uses negligible RAM
    score = predict_prob([text])[0]
    
    return jsonify({"toxic_score": float(score)})

if __name__ == "__main__":
    app.run()
