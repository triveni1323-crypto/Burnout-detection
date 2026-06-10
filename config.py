# config.py
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SECRET_KEY = 'burnout-detection-secret-2024'
DEBUG = True
MODEL_PATH = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data')