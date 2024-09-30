# models/database.py

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# Define your models here
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role = db.Column(db.String(10), nullable=False)
    password = db.Column(db.String(60), nullable=False)
    inferences = db.relationship('InferenceResult', backref='user', lazy=True)

class InferenceResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(120), nullable=False)
    original_path = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    inference_details = db.Column(db.JSON, nullable=False)

class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), unique=True)
    mobile_number = db.Column(db.String(20), nullable=True)
    user = db.relationship('User', backref=db.backref('customer_details', uselist=False, cascade='all, delete'))

class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), unique=True)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    phone_number = db.Column(db.String(20), nullable=True)
    service_name = db.Column(db.String(100), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    orders = db.relationship('Order', backref='agent', lazy=True, cascade='all, delete')
    user = db.relationship('User', backref=db.backref('agent_details', uselist=False, cascade='all, delete'))

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'))
    prediction_path = db.Column(db.String(120), db.ForeignKey('inference_result.image_path'))
    original_path = db.Column(db.String(120), db.ForeignKey('inference_result.original_path'))
    status = db.Column(db.String(50), default='assigned')
    latitude = db.Column(db.Float, nullable=True)  # User's latitude when order is placed
    longitude = db.Column(db.Float, nullable=True)  # User's longitude when order is placed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    timezone = db.Column(db.String(50), nullable=True)
    address = db.Column(db.String(255))
    prediction_id = db.Column(db.Integer, db.ForeignKey('inference_result.id'))
