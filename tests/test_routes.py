# tests/test_routes.py
# Integration tests for Flask routes

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
import json

from run import app

class TestFlaskRoutes(unittest.TestCase):

    def setUp(self):
        """Set up test client"""
        app.config['TESTING'] = True
        app.config['DEBUG']   = False
        self.client = app.test_client()

    def test_home_page(self):
        """Test home page loads"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        print("✅ Home page loads (200 OK)")

    def test_predict_page_get(self):
        """Test predict form page loads"""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        print("✅ Predict page loads (200 OK)")

    def test_dashboard_page(self):
        """Test dashboard page loads"""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        print("✅ Dashboard page loads (200 OK)")

    def test_timer_page(self):
        """Test timer page loads"""
        response = self.client.get('/timer')
        self.assertEqual(response.status_code, 200)
        print("✅ Timer page loads (200 OK)")

    def test_predict_post_low_risk(self):
        """Test prediction form submission - low risk"""
        response = self.client.post('/predict', data={
            'typing_speed':       '75',
            'study_hours':        '5',
            'break_count':        '5',
            'sleep_hours':        '8',
            'mood_score':         '9',
            'device_usage_hours': '4',
            'task_completion':    '90',
            'distraction_count':  '2'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Low', response.data)
        print("✅ Low risk form submission works")

    def test_predict_post_high_risk(self):
        """Test prediction form submission - high risk"""
        response = self.client.post('/predict', data={
            'typing_speed':       '15',
            'study_hours':        '12',
            'break_count':        '1',
            'sleep_hours':        '3',
            'mood_score':         '2',
            'device_usage_hours': '12',
            'task_completion':    '20',
            'distraction_count':  '12'
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'High', response.data)
        print("✅ High risk form submission works")

    def test_api_predict_endpoint(self):
        """Test JSON API prediction endpoint"""
        payload = {
            'typing_speed':       75,
            'study_hours':        5,
            'break_count':        5,
            'sleep_hours':        8,
            'mood_score':         9,
            'device_usage_hours': 4,
            'task_completion':    90,
            'distraction_count':  2
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('risk_level', data['result'])
        self.assertIn('confidence', data['result'])
        print(f"✅ API endpoint works: {data['result']['risk_level']}")

    def test_invalid_route(self):
        """Test 404 for invalid routes"""
        response = self.client.get('/invalid-page')
        self.assertEqual(response.status_code, 404)
        print("✅ Invalid route returns 404")


if __name__ == '__main__':
    print("=" * 50)
    print("   BURNOUT FLASK - INTEGRATION TESTS")
    print("=" * 50)
    unittest.main(verbosity=2)