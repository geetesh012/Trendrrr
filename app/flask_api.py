from flask import Flask, request, jsonify
import joblib
from distilbert_embedding import get_bert_embedding

app = Flask(__name__)
clf = joblib.load("app/svm_distilbert_model.pkl")

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "Text missing"}), 400

    embedding = get_bert_embedding(text)
    pred = clf.predict([embedding])[0]
    return jsonify({"sentiment": label_map[pred]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
