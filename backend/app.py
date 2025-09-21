from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os
from model_utils import preprocess_input, run_fraud_checks

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_models', 'model.joblib')
app = Flask(__name__); CORS(app)
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = preprocess_input(data)
    probs = model.predict_proba([X])[0]
    classes = model.classes_.tolist()
    return jsonify({'risk': classes[int(np.argmax(probs))], 
                    'probabilities': {c: float(p) for c,p in zip(classes, probs)}})

@app.route('/fraud-check', methods=['POST'])
def fraud_check():
    if 'file' not in request.files:
        return jsonify({'error':'no file'}),400
    f = request.files['file']
    return jsonify(run_fraud_checks(f))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
