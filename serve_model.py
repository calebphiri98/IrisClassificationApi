
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)

# Allow requests from the React development server
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173"
        ]
    }
})

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'iris_model.pkl'))


@app.route('/')
def home():
    return jsonify({
        "message": "Iris Model API is running!"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = np.array(data['input'], dtype=float).reshape(1, -1)

        if features.shape[1] != 4:
            return jsonify({
                "error": "Please provide exactly four measurements."
            }), 400

        prediction = model.predict(features)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
