# app/model_utils.py
# Handles loading models and making predictions

import joblib
import json
import numpy as np
import os

# Load models once when app starts
BASE = os.path.dirname(os.path.dirname(__file__))

def load_model(model_name='best_model'):
    path = os.path.join(BASE, 'models', f'{model_name}.pkl')
    return joblib.load(path)

def load_scaler():
    path = os.path.join(BASE, 'models', 'scaler.pkl')
    return joblib.load(path)

def load_features():
    path = os.path.join(BASE, 'models', 'features.json')
    with open(path) as f:
        return json.load(f)

def predict_burnout(data: dict, model_name='best_model'):
    """
    Takes a dictionary of student data,
    returns predicted risk level and confidence
    """
    model    = load_model(model_name)
    scaler   = load_scaler()
    features = load_features()

    # Extract values in correct order
    values = np.array([[data[f] for f in features]])

    # Normalize
    values_scaled = scaler.transform(values)

    # Predict
    prediction    = model.predict(values_scaled)[0]
    probabilities = model.predict_proba(values_scaled)[0]

    # Map number to label
    label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = label_map[prediction]

    # Confidence percentage
    confidence = round(max(probabilities) * 100, 2)

    return {
        'risk_level':    risk_level,
        'confidence':    confidence,
        'probabilities': {
            'Low':    round(probabilities[0] * 100, 2),
            'Medium': round(probabilities[1] * 100, 2),
            'High':   round(probabilities[2] * 100, 2)
        }
    }