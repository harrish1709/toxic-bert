import requests

HF_TOKEN = "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"  # get one free at huggingface.co/settings/tokens

@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.json
    text = data.get("text", "")

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/unitary/toxic-bert",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": text},
            timeout=20
        )
        result = response.json()
        toxic_score = 0
        if isinstance(result, list) and len(result) > 0:
            for item in result[0]:
                if item["label"] == "toxic":
                    toxic_score = item["score"]
                    break
        return jsonify({"toxic_score": toxic_score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()

