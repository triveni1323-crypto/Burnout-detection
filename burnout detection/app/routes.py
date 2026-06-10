# app/routes.py
# All URL routes for the Flask application

from flask import render_template, request, jsonify
from app.model_utils import predict_burnout

def register_routes(app):

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        if request.method == 'GET':
            return render_template('predict.html')

        # POST - process form data
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
        """JSON API endpoint for prediction"""
        try:
            data = request.get_json()
            result = predict_burnout(data)
            return jsonify({'success': True, 'result': result})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})