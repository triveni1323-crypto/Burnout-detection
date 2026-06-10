# app/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static')
    
    app.config['SECRET_KEY'] = 'burnout-detection-2024'
    
    return app