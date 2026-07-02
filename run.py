import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'burnout-detection-2024')

# ── Auto Generate Models If Missing ─────────────────────────
if not os.path.exists('models/best_model.pkl'):
    print("⚠️  Models not found — generating now...")
    os.system('python generate_data.py')
    os.system('python preprocess.py')
    os.system('python train_model.py')
    print("✅ Models ready!")

# ── Prediction Function ──────────────────────────────────────
def predict_burnout(data):
    model    = joblib.load('models/best_model.pkl')
    scaler   = joblib.load('models/scaler.pkl')

    with open('models/features.json') as f:
        features = json.load(f)

    df            = pd.DataFrame([data], columns=features)
    values_scaled = scaler.transform(df)
    prediction    = model.predict(values_scaled)[0]
    probs         = model.predict_proba(values_scaled)[0]

    label_map  = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_level = label_map[prediction]
    confidence = round(float(max(probs)) * 100, 2)

    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'probabilities': {
            'Low':    round(float(probs[0]) * 100, 2),
            'Medium': round(float(probs[1]) * 100, 2),
            'High':   round(float(probs[2]) * 100, 2)
        }
    }

# ── Routes ───────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    try:
        data = {
            'typing_speed':       float(request.form['typing_speed']),
            'study_hours':        float(request.form['study_hours']),
            'break_count':        int(request.form['break_count']),
            'sleep_hours':        float(request.form['sleep_hours']),
            'mood_score':         int(request.form['mood_score']),
            'device_usage_hours': float(request.form['device_usage_hours']),
            'task_completion':    float(request.form['task_completion']),
            'distraction_count':  int(request.form['distraction_count'])
        }
        result = predict_burnout(data)
        result['input_data'] = data
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('predict.html', error=str(e))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/timer')
def timer():
    return render_template('timer.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data   = request.get_json()
        result = predict_burnout(data)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ── Start Server ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting Burnout Detection System...")
    print(f"Open browser: http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)