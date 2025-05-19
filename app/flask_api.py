from flask import Flask, request, jsonify
import joblib
import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel

app = Flask(__name__)

# Load tokenizer and BERT model once (for sentiment)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load sentiment model (SVM using BERT embeddings)
MODEL_DIR = os.path.dirname(__file__)
try:
    sentiment_model = joblib.load(os.path.join(MODEL_DIR, "svm_distilbert_model.pkl"))
except Exception as e:
    sentiment_model = None
    app.logger.error(f"Failed to load sentiment model: {e}", exc_info=True)

# Load trend prediction model (sklearn + vectorizer)
try:
    trend_model = joblib.load(os.path.join(MODEL_DIR, "trend_lr_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
except Exception as e:
    trend_model = None
    vectorizer = None
    app.logger.error(f"Failed to load trend model or vectorizer: {e}", exc_info=True)

# Label map for sentiment
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    if cls_embedding.ndim == 1:
        cls_embedding = cls_embedding.reshape(1, -1)
    return cls_embedding

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    try:
        if sentiment_model is None:
            return jsonify({"error": "Sentiment model not loaded"}), 500

        data = request.get_json() or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text required"}), 400

        emb = get_bert_embedding(text)  # Shape: (1, 768)
        pred = sentiment_model.predict(emb)[0]

        return jsonify({"sentiment": label_map.get(pred, "Unknown")})

    except Exception as e:
        app.logger.error(f"Error in /predict_sentiment: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/predict_trend", methods=["POST"])
def predict_trend():
    try:
        if trend_model is None or vectorizer is None:
            return jsonify({"error": "Trend model or vectorizer not loaded"}), 500

        data = request.get_json() or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text required"}), 400

        X = vectorizer.transform([text])
        prediction = trend_model.predict(X)[0]
        confidence = float(trend_model.predict_proba(X)[0].max()) if hasattr(trend_model, "predict_proba") else None

        return jsonify({
            "will_trend": bool(prediction),
            "confidence": round(confidence, 3) if confidence is not None else None
        })

    except Exception as e:
        app.logger.error(f"Error in /predict_trend: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
