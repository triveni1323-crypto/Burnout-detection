# tests/test_model.py
# Unit tests for ML model predictions

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import json
import numpy as np
import pandas as pd
import unittest

class TestBurnoutModel(unittest.TestCase):

    def setUp(self):
        """Load models before each test"""
        self.model   = joblib.load('models/best_model.pkl')
        self.scaler  = joblib.load('models/scaler.pkl')
        with open('models/features.json') as f:
            self.features = json.load(f)

    def test_model_loads(self):
        """Test model loads successfully"""
        self.assertIsNotNone(self.model)
        print("✅ Model loads successfully")

    def test_scaler_loads(self):
        """Test scaler loads successfully"""
        self.assertIsNotNone(self.scaler)
        print("✅ Scaler loads successfully")

    def test_low_risk_prediction(self):
        """Test low risk student data predicts Low"""
        data = {
            'typing_speed':       75,
            'study_hours':        5,
            'break_count':        5,
            'sleep_hours':        8,
            'mood_score':         9,
            'device_usage_hours': 4,
            'task_completion':    90,
            'distraction_count':  2
        }
        df     = pd.DataFrame([data], columns=self.features)
        scaled = self.scaler.transform(df)
        pred   = self.model.predict(scaled)[0]
        label  = {0:'Low', 1:'Medium', 2:'High'}[pred]
        self.assertEqual(label, 'Low')
        print(f"✅ Low risk prediction: {label}")

    def test_high_risk_prediction(self):
        """Test high risk student data predicts High"""
        data = {
            'typing_speed':       10,
            'study_hours':        14,
            'break_count':        0,
            'sleep_hours':        2,
            'mood_score':         1,
            'device_usage_hours': 14,
            'task_completion':    10,
            'distraction_count':  15
        }
        df     = pd.DataFrame([data], columns=self.features)
        scaled = self.scaler.transform(df)
        pred   = self.model.predict(scaled)[0]
        label  = {0:'Low', 1:'Medium', 2:'High'}[pred]
        self.assertEqual(label, 'High')
        print(f"✅ High risk prediction: {label}")

    def test_medium_risk_prediction(self):
        """Test medium risk student data predicts Medium"""
        data = {
            'typing_speed':       45,
            'study_hours':        8,
            'break_count':        3,
            'sleep_hours':        6,
            'mood_score':         5,
            'device_usage_hours': 8,
            'task_completion':    60,
            'distraction_count':  5
        }
        df     = pd.DataFrame([data], columns=self.features)
        scaled = self.scaler.transform(df)
        pred   = self.model.predict(scaled)[0]
        label  = {0:'Low', 1:'Medium', 2:'High'}[pred]
        self.assertEqual(label, 'Medium')
        print(f"✅ Medium risk prediction: {label}")

    def test_prediction_output_format(self):
        """Test prediction returns valid label 0, 1 or 2"""
        data = {
            'typing_speed':       50,
            'study_hours':        7,
            'break_count':        3,
            'sleep_hours':        6,
            'mood_score':         5,
            'device_usage_hours': 7,
            'task_completion':    60,
            'distraction_count':  5
        }
        df     = pd.DataFrame([data], columns=self.features)
        scaled = self.scaler.transform(df)
        pred   = self.model.predict(scaled)[0]
        self.assertIn(pred, [0, 1, 2])
        print(f"✅ Prediction format valid: {pred}")

    def test_probability_sum(self):
        """Test probabilities sum to 1"""
        data = {
            'typing_speed':       50,
            'study_hours':        7,
            'break_count':        3,
            'sleep_hours':        6,
            'mood_score':         5,
            'device_usage_hours': 7,
            'task_completion':    60,
            'distraction_count':  5
        }
        df     = pd.DataFrame([data], columns=self.features)
        scaled = self.scaler.transform(df)
        probs  = self.model.predict_proba(scaled)[0]
        total  = round(sum(probs), 5)
        self.assertEqual(total, 1.0)
        print(f"✅ Probabilities sum to 1: {total}")

    def test_feature_count(self):
        """Test correct number of features"""
        self.assertEqual(len(self.features), 8)
        print(f"✅ Feature count correct: {len(self.features)}")

    def test_batch_prediction(self):
        """Test multiple predictions at once"""
        batch = pd.DataFrame([
            {'typing_speed':75,'study_hours':5,'break_count':5,'sleep_hours':8,'mood_score':9,'device_usage_hours':4,'task_completion':90,'distraction_count':2},
            {'typing_speed':10,'study_hours':14,'break_count':0,'sleep_hours':2,'mood_score':1,'device_usage_hours':14,'task_completion':10,'distraction_count':15},
            {'typing_speed':45,'study_hours':8,'break_count':3,'sleep_hours':6,'mood_score':5,'device_usage_hours':8,'task_completion':60,'distraction_count':5},
        ], columns=self.features)
        scaled = self.scaler.transform(batch)
        preds  = self.model.predict(scaled)
        self.assertEqual(len(preds), 3)
        print(f"✅ Batch prediction works: {[{0:'Low',1:'Medium',2:'High'}[p] for p in preds]}")

    def test_confidence_range(self):
        """Test confidence is between 0 and 100"""
        data = {
            'typing_speed':       75,
            'study_hours':        5,
            'break_count':        5,
            'sleep_hours':        8,
            'mood_score':         9,
            'device_usage_hours': 4,
            'task_completion':    90,
            'distraction_count':  2
        }
        df         = pd.DataFrame([data], columns=self.features)
        scaled     = self.scaler.transform(df)
        probs      = self.model.predict_proba(scaled)[0]
        confidence = round(max(probs) * 100, 2)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 100)
        print(f"✅ Confidence in valid range: {confidence}%")


if __name__ == '__main__':
    print("=" * 50)
    print("   BURNOUT MODEL - UNIT TESTS")
    print("=" * 50)
    unittest.main(verbosity=2)