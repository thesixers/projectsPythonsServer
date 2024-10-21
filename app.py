from flask import Flask, request, jsonify
import numpy as np
from model import create_model, train_model, load_model, predict, load_data, save_model
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import logging

app = Flask(__name__)

# Initialize variables for the model and scaler
model = None
scaler = None

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_or_train_model():
    global model, scaler
    model_path = './saved_model.keras'

    # Initialize the StandardScaler for feature normalization
    scaler = StandardScaler()  # Ensure scaler is instantiated here

    try:
        model = load_model(model_path)  # Try loading the model
        print("Model loaded successfully.")
        
        # Load data to fit the scaler after loading the model
        features, _ = load_data()  # Load features only
        scaler.fit(features)  # Fit the scaler to the training data

    except Exception as e:
        print(f"Error loading model: {e}")
        # Train the model if not already trained
        features, labels = load_data()  # Load both features and labels
        model = create_model()
        model = train_model(model, features, labels)
        save_model(model, model_path)
        print("Model trained and saved.")

        # Fit the scaler to the training data after training the model
        scaler.fit(features)  # Fit the scaler to the training data


@app.route('/')
def home():
    return "Hello, Flask!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict_health():
    data = request.json
    weight = data.get('weight')
    sleep_hours = data.get('sleepHours')
    bpm = data.get('bpm')
    systolic = data.get('systolic')
    diastolic = data.get('diastolic')

    # Ensure all inputs are provided
    if None in [weight, sleep_hours, bpm, systolic, diastolic]:
        return jsonify({'error': 'All input features must be provided.'}), 400

    try:
        # Create input data for prediction
        input_data = np.array([[weight, sleep_hours, bpm, systolic, diastolic]])
        
        # Scale input data using the trained scaler
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = predict(model, scaled_input)

        # Threshold the output to determine the class (0 or 1)
        health_status = 1 if prediction >= 0.5 else 0

        return jsonify({
            'prediction': float(prediction),
            'health_status': 'Hypertension' if health_status == 1 else 'Normal'
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed.'}), 500

if __name__ == '__main__':
    load_or_train_model()  # Load or train the model when starting the server
    app.run(debug=True, port=5000)
