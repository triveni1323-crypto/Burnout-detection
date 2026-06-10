from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import os

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

app.config['SECRET_KEY'] = 'burnout-detection-2024'

# ── Load Models ──────────────────────────────
BASE = os.path.dirname(__file__)

def predict_burnout(data):
    model    = joblib.load(os.path.join(BASE, 'models', 'best_model.pkl'))
    scaler   = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))
    
    with open(os.path.join(BASE, 'models', 'features.json')) as f:
        features = json.load(f)

    values        = np.array([[data[f] for f in features]])
    values_scaled = scaler.transform(values)
    prediction    = model.predict(values_scaled)[0]
    probs         = model.predict_proba(values_scaled)[0]

    label_map  = {0: 'Low', 1: 'Medium', 2: 'High'}

    return {
        'risk_level': label_map[prediction],
        'confidence': round(max(probs) * 100, 2),
        'probabilities': {
            'Low':    round(probs[0] * 100, 2),
            'Medium': round(probs[1] * 100, 2),
            'High':   round(probs[2] * 100, 2)
        }
    }

# ── Routes ───────────────────────────────────
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
@app.route('/api/predict', methods=['POST'])
def api_predict():
     try:
        data   = request.get_json()
        result = predict_burnout(data)
        return jsonify({'success': True, 'result': result})
     except Exception as e:
         return jsonify({'success': False, 'error': str(e)})
@app.route('/timer')
def timer():
    return render_template('timer.html')

# ── Start Server ──────────────────────────
if __name__ == '__main__':
    print("Starting Burnout Detection System...")
    print("Open browser: http://127.0.0.1:5000")
    import os
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port, debug=False)