from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import numpy as np
import io
import uuid
from datetime import datetime
from pytz import timezone
from flask import current_app
import os
import requests
import faiss
from flask_socketio import SocketIO, emit
from math import radians, sin, cos, sqrt, atan2
import logging
from sqlalchemy import event
logging.basicConfig(level=logging.DEBUG)
def basename_filter(path):
    return os.path.basename(path)
import torch
import numpy as np
import cv2
from base64 import b64encode
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import re
import statistics
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from matplotlib.patches import Rectangle
import json
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv() 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.jinja_env.filters['basename'] = basename_filter
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, async_mode='threading')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
# OPENAI_API_KEY = "sk-proj-your_openai_api_key"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
gmaps_api_key = os.getenv('GoogleMaps_API_KEY')
gsearch_api_key = os.getenv('GoogleSearch_API_KEY')
gsearch_engine_id = os.getenv('GoogleSearch_engine_id')
modelName = 'gpt-4o-mini'
if modelName.startswith("gpt"):
    LLM = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=modelName, temperature=0.2)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
else:
    LLM = Ollama(model=modelName)
    embeddings = OllamaEmbeddings(model=modelName, temperature=0.2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Initializing model: {modelName}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Initializing model: {modelName}")


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

with app.app_context():
    db.create_all()

def create_agent_entry(mapper, connection, target):
    if target.role == 'agent':
        # Creating a new Agent instance related to the new User
        new_agent = Agent(user_id=target.id)
        # Use the connection to directly insert the new Agent without affecting the session transaction
        connection.execute(
            Agent.__table__.insert(),
            {"user_id": new_agent.user_id}
        )
        # Do not use commit here; let the calling transaction handle it

event.listen(User, 'after_insert', create_agent_entry)

def delete_agent_entry(mapper, connection, target):
    if target.role == 'agent':
        # Deleting the Agent entry linked to the User
        connection.execute(
            Agent.__table__.delete().where(Agent.user_id == target.id)
        )

event.listen(User, 'after_insert', create_agent_entry)
event.listen(User, 'after_delete', delete_agent_entry)

def create_customer_entry(mapper, connection, target):
    if target.role == 'user':
        # Creating a new Customer instance related to the new User
        new_customer = Customer(user_id=target.id)
        # Use the connection to directly insert the new Customer without affecting the session transaction
        connection.execute(
            Customer.__table__.insert(),
            {"user_id": new_customer.user_id}
        )
        # Do not use commit here; let the calling transaction handle it

event.listen(User, 'after_insert', create_customer_entry)

def delete_customer_entry(mapper, connection, target):
    if target.role == 'user':
        connection.execute(
            Customer.__table__.delete().where(Customer.user_id == target.id)
        )
        # Commit is managed by the calling transaction

event.listen(User, 'after_delete', delete_customer_entry)


def calculate_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    print('************************')
    print(f'Distance calculated between user location and agent is {distance} kilometers')
    print('************************')
    
    return distance

def find_nearest_agent(user_latitude, user_longitude):
    nearest_distance = float('inf')
    nearest_agent = None
    agents = Agent.query.all()
    for agent in agents:
        if agent.latitude is not None and agent.longitude is not None:
            distance = calculate_distance(user_latitude, user_longitude, agent.latitude, agent.longitude)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_agent = agent
    print('************************')
    print(f'Nearest agent distance is {nearest_distance} kilometers, and nearest agent is {nearest_agent}')
    print('************************')
    return nearest_agent

def reverse_geocode(lat, lng):
    
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lng}",
        "key": gmaps_api_key
    }
    response = requests.get(base_url, params=params)
    results = response.json()['results']
    if results:
        return results[0]['formatted_address']
    return None

import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    return obj


@app.template_filter('to_user_timezone')
def to_user_timezone(value, tz_string):
    try:
        user_tz = timezone(tz_string)
        return value.astimezone(user_tz)
    except Exception as e:
        current_app.logger.error(f"Error converting timezone: {str(e)}")
        return value  # fallback to the original time if the timezone conversion fails
    

@app.context_processor
def inject_user():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        return {'user': user}
    return {'user': None}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/create_user', methods=['GET', 'POST'])
def create_user():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        email = request.form['email']
        role = request.form['role']
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('admin_dashboard'))
        new_user = User(username=username, email=email, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('User created successfully', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('create_user.html')

@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    if session.get('role') != 'admin' or session.get('user_id') == user_id:
        return "Unauthorized", 403
    user = User.query.get_or_404(user_id)
    if request.method == 'POST':
        user.username = request.form['username']
        user.email = request.form['email']
        # Update the password only if a new one is provided:
        if request.form['password']:
            user.password = generate_password_hash(request.form['password'])
        db.session.commit()
        flash('User updated successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('edit_user.html', user=user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        role = request.form['role']
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('login'))
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('User registered successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        user = User.query.filter_by(username=username, role=role).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['role'] = user.role
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'agent':
                return redirect(url_for('agent_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            return 'Login Unsuccessful. Please check username and password'
    return render_template('login.html')


@app.route('/delete_user/<int:user_id>', methods=['GET'])
def delete_user(user_id):
    if session.get('user_id') == user_id:
        flash('You cannot delete your own account.', 'error')
        return redirect(url_for('admin_dashboard'))
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/user_profile', methods=['GET', 'POST'])
def user_profile():
    if 'user_id' not in session:
        flash("Please log in to update your profile.", "warning")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get_or_404(user_id)
    customer = Customer.query.filter_by(user_id=user_id).first()

    if request.method == 'POST':
        user.email = request.form['email']
        customer.mobile_number = request.form['mobile_number']
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('user_dashboard'))

    return render_template('user_profile.html', user=user, customer=customer)


@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    if user:
        # Fetch orders with the associated agent and inference results
        inferences = (
            db.session.query(Order, Agent, InferenceResult, User.username.label('agent_username'))
            .join(Agent, Order.agent_id == Agent.id)
            .join(User, Agent.user_id == User.id)
            .outerjoin(InferenceResult, Order.prediction_id == InferenceResult.id)  # Join with inference_result
            .filter(Order.user_id == user.id)
            .all()
        )

        # Prepare data to pass to the template
        inference_data = []
        for order, agent, inference_result, agent_username in inferences:
            if inference_result and isinstance(inference_result.inference_details, dict):
                inference_details = inference_result.inference_details  # Already a dict
            else:
                inference_details = {}

            # Extract paths from inference_details
            original_path = inference_details.get('orig_image', '')
            prediction_path = inference_details.get('inf_image', '')

            # Append data for each inference
            inference_data.append({
                'order': order,
                'agent_username': agent_username,
                'original_path': original_path,
                'prediction_path': prediction_path
            })

        return render_template('user_dashboard.html', user=user, inferences=inference_data)
    
    else:
        flash("User not found", "error")
        return redirect(url_for('login'))


@app.route('/agent_dashboard')
def agent_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    agent = Agent.query.filter_by(user_id=user_id).first()
    car_model = ''
    car_year = ''
    
    if agent:
        assigned_orders = Order.query.filter_by(agent_id=agent.id).all()
        
        orders_data = []
        for order in assigned_orders:
            print(f'***********order number assigned to agent is {order.id}*******')
            
            # Check if prediction_id is not None before querying
            inference_result = None
            if order.prediction_id is not None:
                inference_result = InferenceResult.query.filter_by(id=order.prediction_id).first()
            print(f'*********inference result is {inference_result}*********')
            
            hours_since_created = None  # Default to None
            
            if order.created_at and order.timezone:
                order_tz = timezone(order.timezone)
                now_in_order_tz = datetime.now(order_tz)
                
                created_at_tz_aware = order.created_at.replace(tzinfo=timezone('UTC')).astimezone(order_tz)
                time_diff = now_in_order_tz - created_at_tz_aware
                hours_since_created = time_diff.total_seconds() // 3600

            if inference_result and isinstance(inference_result.inference_details, dict):
                inference_details = inference_result.inference_details  # It's already a dictionary
            else:
                inference_details = {}  # Handle the case where inference_details is not found or is not a dict

            image_path = inference_details.get('inf_image', '')
            car_model = inference_details.get('car_model')
            car_year = inference_details.get('car_year')
            # Get image path from inference_result
            # image_path = inference_result.prediction_path if inference_result else None
            
            # Append order data to the list
            orders_data.append({
                'id': order.id,
                'status': order.status,
                'image_path': image_path,
                'hours_since_created': hours_since_created  # Could be None if not available
            })
        
        print(f'*******orders data is {orders_data}**********')
        return render_template('agent_dashboard.html', orders=orders_data, agent=agent, car_model = str(car_model)+' '+str(car_year))
    else:
        flash("You are not registered as an agent.", "error")
        return redirect(url_for('home'))


@app.route('/agent_profile')
def agent_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    agent = Agent.query.filter_by(user_id=user_id).first()

    if not agent:
        flash("You are not registered as an agent.", "error")
        return redirect(url_for('home'))
    
    return render_template('agent_profile.html', agent=agent, google_maps_api_key=gmaps_api_key)


@app.route('/update_agent_profile', methods=['POST'])
def update_agent_profile():
    if 'user_id' not in session or session.get('role') != 'agent':
        flash("You need to log in as an agent to update your profile.", "warning")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    agent = Agent.query.filter_by(user_id=user_id).first()
    
    if agent:
        # agent.latitude = request.form['latitude']
        # agent.longitude = request.form['longitude']
        agent.phone_number = request.form['phone_number']
        agent.service_name = request.form['service_name']
        agent.address = request.form['address']
        db.session.commit()
        flash('Profile updated successfully!', 'success')
    else:
        flash('No agent profile found.', 'error')
    
    return redirect(url_for('agent_profile'))

@app.route('/update_agent_location', methods=['POST'])
def update_agent_location():
    if 'user_id' not in session or session.get('role') != 'agent':
        return redirect(url_for('login'))
    user_id = session['user_id']
    agent = Agent.query.filter_by(user_id=user_id).first()
    if agent:
        agent.latitude = request.form['latitude']
        agent.longitude = request.form['longitude']
        db.session.commit()
        flash('Location updated successfully!', 'success')
    else:
        new_agent = Agent(user_id=user_id, latitude=request.form['latitude'], longitude=request.form['longitude'])
        db.session.add(new_agent)
        db.session.commit()
        flash('Location set successfully!', 'success')
    return redirect(url_for('agent_dashboard'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    admin = User.query.get(session['user_id'])
    users = User.query.all()
    return render_template('admin_dashboard.html', admin=admin, users=users)

@app.route('/order_details/<int:order_id>')
def view_order_details(order_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    order = Order.query.get_or_404(order_id)
    agent = Agent.query.filter_by(user_id=session['user_id']).first()
    if not agent or order.agent_id != agent.id:
        flash("You are not authorized to view this page.", "error")
        return redirect(url_for('agent_dashboard'))
    
    inference_result = InferenceResult.query.filter_by(id=order.prediction_id).first()
    if inference_result and isinstance(inference_result.inference_details, dict):
        inference_details = inference_result.inference_details  # It's already a dictionary
    else:
        inference_details = {} 
    original_path = inference_details.get('orig_image', '')
    prediction_path = inference_details.get('inf_image', '')
    car_model = inference_details.get('car_model')
    car_year = inference_details.get('car_year')
    damage_details = inference_details.get('damage_info')
    customer = Customer.query.filter_by(user_id=order.user_id).first()
    customer_phone = customer.mobile_number if customer else 'Customer phone number not avaialable, please contact via email'
    
    user = User.query.get(customer.user_id) if customer else None
    customer_email = user.email if user else 'Customer email not available'

    return render_template('order_details.html', google_maps_api_key=gmaps_api_key, order=order, agent=agent, customer_phone=customer_phone, customer_email=customer_email, original_path=original_path, prediction_path=prediction_path, car_model = str(car_model)+' '+str(car_year), damage_details=damage_details)

@app.route('/user_order_details/<int:order_id>')
def user_order_details(order_id):
    order = Order.query.get_or_404(order_id)

    if 'user_id' not in session or session['user_id'] != order.user_id:
        flash("Unauthorized access.", "error")
        return redirect(url_for('login'))

    agent = Agent.query.get(order.agent_id)
    agent_user = User.query.get(agent.user_id) if agent else None
    agent_phone = agent.phone_number if agent else 'N/A'
    agent_email = agent_user.email if agent_user else 'No email found'
    agent_username = agent_user.username if agent_user else 'N/A'

    inference_result = InferenceResult.query.filter_by(id=order.prediction_id).first()

    if inference_result and isinstance(inference_result.inference_details, dict):
        inference_details = inference_result.inference_details
    else:
        inference_details = {}

    original_path = inference_details.get('orig_image', '')
    prediction_path = inference_details.get('inf_image', '')

    # Extracting detailed damage information
    damage_info = inference_details.get('damage_info', {})
    chosen_damages = inference_details.get('chosen_damages', [])
    repair_cost = inference_details.get('repair_cost', 0)

    return render_template(
        'user_order_details.html',
        google_maps_api_key=gmaps_api_key,
        order=order,
        agent_username=agent_username,
        agent_phone=agent_phone,
        agent_email=agent_email,
        original_path=original_path,
        prediction_path=prediction_path,
        damage_info=damage_info,
        chosen_damages=chosen_damages,  # Pass the detailed breakdown
        repair_cost=repair_cost  # Pass the total repair cost
    )


@app.route('/accept_order/<int:order_id>', methods=['POST'])
def accept_order(order_id):
    order = Order.query.get_or_404(order_id)
    order.status = 'accepted'
    db.session.commit()
    flash('Order accepted successfully.', 'success')
    return redirect(url_for('agent_dashboard'))

@app.route('/decline_order/<int:order_id>', methods=['POST'])
def decline_order(order_id):
    order = Order.query.get_or_404(order_id)
    next_agent = find_next_nearest_agent(order.latitude, order.longitude, order.agent_id)
    if next_agent:
        order.agent_id = next_agent.id
        db.session.commit()
        flash('Order declined and reassigned.', 'info')
    else:
        order.status = 'declined'
        db.session.commit()
        flash('Order declined and no other agents available.', 'error')
    return redirect(url_for('agent_dashboard'))

@app.route('/mark_finished/<int:order_id>', methods=['POST'])
def mark_finished(order_id):
    order = Order.query.get_or_404(order_id)
    if order and order.status == 'accepted':
        order.status = 'PaymentRequested'
        db.session.commit()
        flash('Payment requested successfully.', 'info')
    return redirect(url_for('agent_dashboard'))

@app.route('/complete_payment/<int:order_id>', methods=['POST'])
def complete_payment(order_id):
    order = Order.query.get_or_404(order_id)
    if order and order.status == 'PaymentRequested':
        order.status = 'Completed'
        db.session.commit()
        flash('Payment completed and order marked as completed.', 'success')
    return redirect(url_for('user_dashboard'))

# Inferencing Model


CONFIG_FILE = 'carDDModel/dcn_plus_cfg_small.py'
CHECKPOINT_FILE = 'carDDModel/checkpoint.pth'
DEVICE = 'cuda:0'
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
transform = T.ToTensor()


def fetch_search_results(api_key, search_engine_id, query, num_results=4):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}&num={num_results}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('items', [])
    else:
        print(f"Error: {response.status_code}")
        return []

def fetch_webpage_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return None

def extract_price_from_webpage(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    price_pattern = re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
    prices = price_pattern.findall(soup.get_text())
    prices = [float(price.replace('$', '').replace(',', '')) for price in prices]
    return prices

def extract_prices_from_snippets(snippets):
    price_pattern = re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
    prices = []
    for snippet in snippets:
        found_prices = price_pattern.findall(snippet)
        prices.extend([float(price.replace('$', '').replace(',', '')) for price in found_prices])
    return prices

def get_car_price_from_web(api_key, search_engine_id, query, num_results=4):
    search_results = fetch_search_results(api_key, search_engine_id, query, num_results)
    all_prices = []
    for result in search_results:
        url = result['link']
        html_content = fetch_webpage_content(url)
        if html_content:
            prices = extract_price_from_webpage(html_content)
            all_prices.extend(prices)
    if all_prices:
        avg_price = sum(all_prices) / len(all_prices)
        return avg_price
    else:
        return None

@app.route('/fetch-car-price', methods=['POST'])
def fetch_car_price():
    car_model = request.form.get('car_model')
    car_year = request.form.get('car_year')
    # session['car_model'] = car_model
    # session['car_year'] = car_year    
    query = f'{car_year} {car_model} MSRP'
    avg_price = get_car_price_from_web(gsearch_api_key, gsearch_engine_id, query, num_results=2)
    
    if avg_price:
        session['car_price'] = f'{avg_price:.2f}'
        car_price = f'{avg_price:.2f}'
        # print(f'************car price fetched in fetch_car_price is {car_price}*********')
    else:
        car_price = '0.00'  # Default to '0.00' if the price is not found
    
    # Render the template with the fetched car price
    return render_template('upload.html', car_price=car_price, car_model=car_model, car_year=car_year, step='price_fetched', google_maps_api_key=gmaps_api_key)


def get_best_vehicle_box(det_output):
    max_score = 0
    max_bbox = None
    vehicle_classes = [2, 3, 7, 8]  # Car, truck, bus in MS COCO
    # print(f"Number of bounding boxes detected: {len(det_output['boxes'])}")
    for i in range(len(det_output['boxes'])):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        label = det_output['labels'][i]
        print(f"Detected object class: {label}, score: {score}, bbox: {bbox}")
        if label in vehicle_classes and score > max_score:
            max_bbox = bbox
            max_score = score
    # print(f"Selected bounding box: {max_bbox}")
    return max_bbox

def adjust_bbox(bbox, img_width, img_height, margin=0.1):
    """
    Adjust the bounding box to include a margin around it.

    Parameters:
    - bbox: [x1, y1, x2, y2] coordinates of the bounding box
    - img_width: width of the original image
    - img_height: height of the original image
    - margin: percentage to increase the bounding box size by (default 10%)

    Returns:
    - adjusted_bbox: adjusted bounding box with the margin applied
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Calculate the margin
    x_margin = int(width * margin)
    y_margin = int(height * margin)

    # Apply the margin, ensuring the bounding box stays within the image boundaries
    new_x1 = max(0, x1 - x_margin)
    new_y1 = max(0, y1 - y_margin)
    new_x2 = min(img_width, x2 + x_margin)
    new_y2 = min(img_height, y2 + y_margin)

    return [new_x1, new_y1, new_x2, new_y2]


def process_damage_analysis(result, classification_dict, confidence_threshold=0.5):
    """
    Process the results from the damage detection model and print human-readable damage information.

    Parameters:
    - result: The output from the inference_detector function, a tuple containing bounding boxes and segmentation data.
    - classification_dict: A dictionary mapping classification labels to human-readable damage types.
    - confidence_threshold: The minimum confidence score for considering a damage detection valid.

    Returns:
    - A dictionary containing human-readable damage information.
    - A dictionary containing the damage masks for each valid detection.
    """
    damage_info = {classification_dict[i]: [] for i in classification_dict.keys()}
    damage_masks = {classification_dict[i]: [] for i in classification_dict.keys()}

    for classification_label, bboxes in enumerate(result[0], 1):
        damage_type = classification_dict[classification_label]

        for i, bbox in enumerate(bboxes):
            score = bbox[4]
            if score >= confidence_threshold:
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)
                
                damage_info[damage_type].append({
                    'index': i + 1,
                    'area': area,
                    'score': score
                })
                
                # Only add the mask if it exists and the index is valid
                if len(result[1]) > classification_label and len(result[1][classification_label]) > i:
                    damage_masks[damage_type].append(result[1][classification_label][i])

    for damage_type, damages in damage_info.items():
        if damages:
            print(f"Detected {damage_type}:")
            for damage in damages:
                if damage_type.lower() == 'tire_flat':
                    print(f"- {damage_type.capitalize()} with confidence score of: {damage['score']:.2f}")
                else:
                    print(f"- {damage_type.capitalize()} {damage['index']} area is: {damage['area']:.2f} "
                          f"with confidence score of: {damage['score']:.2f}")
        else:
            print(f"No {damage_type} detected.")
    
    return damage_info, damage_masks

def calculate_repair_cost_analysis(damage_info, car_value, car_age, damage_factors, image_size):
    """
    Calculate the estimated repair cost based on the detected damage, car value, and car age.

    Parameters:
    - damage_info: A dictionary containing information about detected damage.
    - car_value: The initial value of the car.
    - car_age: The age of the car (capped at 10 years).
    - damage_factors: A dictionary containing factors for different types of damage.
    - image_size: A tuple (width, height) representing the size of the image in pixels.

    Returns:
    - The estimated repair cost.
    - A detailed breakdown of the selected damages and their costs.
    """
    repair_cost = 0

    # Calculate the total image area based on its dimensions
    total_image_area = image_size[0] * image_size[1]  # width * height in pixels

    # Cap the age of the car to a maximum of 10 years
    car_age = min(car_age, 10)
    print(f'********car age calculated is {car_age}*******')
    age_factor = max(0.5/100, (car_age / 10.0))  # Simple linear depreciation based on age
    print(f'********age_factor calculated is {age_factor}*******')
    chosen_damages = []

    for damage_type, damages in damage_info.items():
        factor = damage_factors[damage_type]
        for damage in damages:
            area_ratio = damage['area'] / total_image_area
            cost = area_ratio * damage['score'] * car_value * factor * age_factor
            repair_cost += cost

            chosen_damages.append({
                'damage_type': damage_type,
                'damage_index': damage['index'],
                'area': damage['area'],
                'score': damage['score'],
                'cost': cost
            })

    print(f"Total estimated repair cost: ${repair_cost:.2f}")
    print("\nDetailed Damage Cost Breakdown:")
    for damage in chosen_damages:
        print(f"- {damage['damage_type'].capitalize()} {damage['damage_index']} (Area: {damage['area']:.2f}, "
              f"Score: {damage['score']:.2f}): ${damage['cost']:.2f}")

    return repair_cost, chosen_damages

def visualize_damage_and_costs(damage_info, repair_cost, chosen_damages):
    """
    Visualize the damage types and their counts using a bar plot,
    and visualize the cost breakdown by damage type using a pie chart.
    """
    # Prepare data for the bar plot
    damage_types = []
    damage_counts = []
    
    for damage_type, damages in damage_info.items():
        if damages:
            damage_types.append(damage_type.capitalize())
            damage_counts.append(len(damages))
    
    # Create the figure and subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar Plot: Count of Each Damage Type
    ax[0].bar(damage_types, damage_counts, color='skyblue')
    ax[0].set_title('Count of Each Damage Type')
    ax[0].set_xlabel('Damage Type')
    ax[0].set_ylabel('Count')
    
    # Prepare data for the pie chart
    damage_costs = {}
    
    for damage in chosen_damages:
        damage_type = damage['damage_type'].capitalize()
        cost = damage['cost']
        if damage_type in damage_costs:
            damage_costs[damage_type] += cost
        else:
            damage_costs[damage_type] = cost
    
    # Pie Chart: Repair Cost Breakdown by Damage Type
    ax[1].pie(damage_costs.values(), labels=damage_costs.keys(), autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax[1].set_title(f'Repair Cost Breakdown by Damage Type (Total: ${repair_cost:.2f})')
    
    # Return the figure instead of showing it
    return fig


@app.route('/check-car-damages', methods=['GET', 'POST'])
def check_car_damages():
    if request.method == 'GET':
        # Fetch the values from session and render the page
        car_price = request.args.get('car_price', '')
        car_year = request.args.get('car_year', '')
        car_model = request.args.get('car_model', '')
        print(f'car price in check_car_damage get method is {car_price}*********')
        return render_template('upload.html', car_price=car_price, car_year=car_year, car_model=car_model, google_maps_api_key=gmaps_api_key)

        
    elif request.method == 'POST':
        car_price = request.form.get('car_price')
        car_year = request.form.get('car_year')
        car_model = request.form.get('car_model')
        file = request.files['file']

        if file and file.filename.endswith('.jpg'):
            try:
                image = Image.open(file.stream)
                # Object detection pipeline
                # det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
                # transform = T.ToTensor()
                img_tensor = transform(image).unsqueeze(0).cuda()
                det_output = det_model(img_tensor)[0]

                # Get the best bounding box for vehicle
                best_bbox = get_best_vehicle_box(det_output)
                if best_bbox is not None:
                    bbox_np = best_bbox.detach().cpu().numpy().astype(int)
                    # print(f"Bounding box before adjustment: {bbox_np}")

                    # Adjust the bounding box size by 10%
                    img_width, img_height = image.size
                    adjusted_bbox = adjust_bbox(bbox_np, img_width, img_height, margin=0.1)
                    # print(f"Bounding box after adjustment: {adjusted_bbox}")

                    # Crop the image to the adjusted bounding box
                    cropped_img = image.crop(adjusted_bbox)

                    # Save the original and cropped images
                    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
                    orig_filename = f'{timestamp}_original.jpg'
                    cropped_filename = f'{timestamp}_cropped.jpg'
                    orig_filepath = os.path.join('results', orig_filename).replace('\\', '/')
                    cropped_filepath = os.path.join('results', cropped_filename).replace('\\', '/')
                    print(f'********cropped_filepath is {cropped_filepath}********')

                    # Store paths in session
                    # session['orig_image_path'] = orig_filepath
                    # session['cropped_image_path'] = cropped_filepath

                    image.save(os.path.join(app.static_folder, orig_filepath))
                    Image.fromarray(np.array(cropped_img)).save(os.path.join(app.static_folder, cropped_filepath))

                    # Render the confirmation page with original and cropped images
                    return render_template('upload.html', orig_image=orig_filepath, cropped_image=cropped_filepath,
                                       step='confirm', car_price=car_price, car_year=car_year, car_model=car_model, google_maps_api_key=gmaps_api_key)

                else:
                    flash("No vehicle detected in the image. Please upload a clearer image.", 'error')
                    return redirect(url_for('check_car_damages'))

            except Exception as e:
                print("Error processing image in check_car_damage:", e)
                flash("Error processing image.", 'error')
                return redirect(url_for('check_car_damages'))
        else:
            flash("Unsupported file type or no file found", 'error')
            return redirect(url_for('check_car_damages'))


@app.route('/run-damage-inference', methods=['POST'])
def run_damage_inference():
    try:
        # Retrieve form data
        cropped_image_path = request.form.get('cropped_image_path')
        orig_image_path = request.form.get('orig_image_path')
        car_price = request.form.get('car_price')
        car_year = request.form.get('car_year')
        car_model = request.form.get('car_model')
        
        # Debugging: Print retrieved values
        print(f'*** car_price: {car_price}, car_year: {car_year}, car_model: {car_model} ***')
        
        # Validate retrieved values
        if not all([cropped_image_path, orig_image_path, car_price, car_year, car_model]):
            flash("Missing required information for damage inference.", 'error')
            return redirect(url_for('check_car_damages'))
        
        # Convert car_price and car_year to appropriate types
        car_price = float(car_price)
        car_year = int(car_year)
        
        # Load cropped image
        cropped_image_full_path = os.path.join(app.static_folder, cropped_image_path)
        cropped_image = Image.open(cropped_image_full_path)
        cropped_img_np = np.array(cropped_image)
        
        # Perform damage detection inference
        damage_result = inference_detector(model, cropped_img_np)
        
        # Define classification mapping
        classification_dict = {1: 'dent', 2: 'scratch', 3: 'crack', 4: 'glass_shatter', 5: 'lamp_broken', 6: 'tire_flat'}
        
        # Process damage results
        damage_info, damage_masks = process_damage_analysis(damage_result, classification_dict)
        
        # Define damage factors
        damage_factors = {'dent': 0.3, 'scratch': 0.2, 'crack': 0.3, 'glass_shatter': 0.35, 'lamp_broken': 0.4, 'tire_flat': 0.1}
        
        # Calculate repair cost
        image_size = cropped_image.size
        repair_cost, chosen_damages = calculate_repair_cost_analysis(
            damage_info, car_price, car_year, damage_factors, image_size
        )
        PALETTE = [(255, 182, 193), (0, 168, 225), (0, 255, 0), (128, 0, 128), (255, 255, 0), (227, 0, 57)] 
        final_cropped_result = model.show_result(cropped_img_np, damage_result, show=False, score_thr=0.5, text_color=PALETTE, mask_color=PALETTE, font_size=20)
        # Generate and save inference result image
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        inf_filename = f'{timestamp}_inference.jpg'
        inf_filepath = os.path.join('results', inf_filename).replace('\\', '/')
        print(f'********inf_filepath is {inf_filepath}********')
        # inf_full_path = os.path.join(app.static_folder, inf_filepath)

        Image.fromarray(final_cropped_result).save(os.path.join(app.static_folder, inf_filepath))

        # Visualize and save inference result
        # visualize_damage(cropped_img_np, damage_result, classification_dict, save_path=inf_full_path)
        
        # Pass all required data to the template
        inference_details = {
            'car_price': car_price,
            'car_year': car_year,
            'car_model': car_model,
            'orig_image': orig_image_path,
            'cropped_image': cropped_image_path,
            'inf_image': inf_filepath,
            'repair_cost': repair_cost,
            'damage_info': damage_info,
            'chosen_damages': chosen_damages
        }
        # if 'user_id' not in session:
        #     print("User is not logged in, Entering User id as Blank")
        #     user.id = 999
        # else:
        #     user = User.query.get(session['user_id'])

        inference_details_serializable = {key: convert_to_serializable(value) for key, value in inference_details.items()}
        print(f'Inference details are serialization are {inference_details_serializable}')
        inference_result = InferenceResult(
            user_id=999,  # Assuming you are using Flask-Login for user authentication
            image_path=inf_filepath,
            original_path=orig_image_path,
            inference_details=inference_details_serializable
        )
        
        # Add and commit the inference result to the database
        db.session.add(inference_result)
        db.session.commit()
        inference_id = inference_result.id
        session['inference_id'] = inference_id
        print(f'************Generated inference id is {inference_id}****************')

        return render_template(
            'upload.html',
            orig_image=orig_image_path,
            cropped_image=cropped_image_path,
            inf_image=inf_filepath,
            repair_cost=repair_cost,
            damage_info=damage_info,
            chosen_damages=chosen_damages,
            car_price=car_price,
            car_year=car_year,
            car_model=car_model,
            step='result',
            google_maps_api_key=gmaps_api_key
        )
    
    except Exception as e:
        print("Error processing damage inference in run_damage_inference:", e)
        flash("Error processing damage inference.", 'error')
        return redirect(url_for('check_car_damages'))

@app.route('/detailed-analysis', methods=['POST'])
def detailed_analysis():
    try:
        # Retrieve data passed from the previous step
        repair_cost = float(request.form.get('repair_cost'))
        damage_info = eval(request.form.get('damage_info'))  # Convert string representation back to a dictionary
        print(f'************damage info as generated from image are {damage_info}***********************')
        chosen_damages = eval(request.form.get('chosen_damages'))

        # Generate visualizations
        img = visualize_damage_and_costs(damage_info, repair_cost, chosen_damages)
        print(f'Visualize damage function ran succesfully')
        # Convert plot to PNG image and base64 encode it
        buf = BytesIO()
        print(f'Image savefig function next')
        img.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        print(f'Image save function ran sucessfully, next is return function')
        return render_template('detailed_analysis.html', repair_cost=repair_cost, image_base64=image_base64)

    except Exception as e:
        print("Error generating detailed analysis:", e)
        flash("Error generating detailed analysis.", 'error')
        return redirect(url_for('check_car_damages'))

@app.route('/request_repair', methods=['POST'])
def request_repair():
    if 'user_id' not in session:
        flash("Please log in to request repairs.", "warning")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:
        flash("Invalid request. Please ensure a valid session.", "error")
        return redirect(url_for('user_dashboard'))

    if user.role != 'user':
        flash("Please log in as a customer to request repair services.", "error")
        return redirect(url_for('user_dashboard'))

    inference_id = session.get('inference_id')
    print(f'*****************Inference id fetched from session is {inference_id}************')
    # Retrieving form data
    prediction_path = request.form.get('prediction_path')
    original_path = request.form.get('original_path')
    user_latitude = request.form.get('latitude')
    user_longitude = request.form.get('longitude')
    user_timezone = request.form.get('timezone')

    try:
        user_latitude = float(user_latitude)
        user_longitude = float(user_longitude)
    except (TypeError, ValueError):
        flash("Invalid location data. Please ensure it is correctly provided.", "error")
        return redirect(url_for('check_car_damages'))
    
    address = reverse_geocode(user_latitude, user_longitude)
    nearest_agent, min_distance = find_nearest_agent(user_latitude, user_longitude)
    max_distance = 500
    if nearest_agent is None or min_distance > max_distance:
        flash(f"No nearby agents found within (max_distance) km.", "error")
        return redirect(url_for('check_car_damages'))

    # Creating a new order
    new_order = Order(
        user_id=user.id, 
        agent_id=nearest_agent.id, 
        prediction_path=prediction_path,
        original_path=original_path,
        prediction_id=inference_id,
        latitude=user_latitude, 
        longitude=user_longitude, 
        status='assigned',
        timezone=user_timezone,
        address=address
    )
    db.session.add(new_order)
    db.session.commit()
    flash("Repair request submitted successfully!", "success")
    return redirect(url_for('user_dashboard'))

def find_nearest_agent(lat, lon):
    nearest_agent = None
    min_distance = float('inf')
    agents = Agent.query.all()
    for agent in agents:
        if agent.latitude is not None and agent.longitude is not None:
            distance = calculate_distance(lat, lon, agent.latitude, agent.longitude)
        if distance < min_distance:
            min_distance = distance
            nearest_agent = agent
    return nearest_agent, min_distance

def find_next_nearest_agent(lat, lon, exclude_agent_id):
    agents = Agent.query.filter(Agent.id != exclude_agent_id).all()
    nearest_agent = None
    min_distance = float('inf')
    for agent in agents:
        if agent.latitude is not None and agent.longitude is not None:
            distance = calculate_distance(lat, lon, agent.latitude, agent.longitude)
            if distance < min_distance:
                min_distance = distance
                nearest_agent = agent
    return nearest_agent

# LLM Model

# WebSocket status message function
def send_status(message):
    socketio.emit('status', message, namespace='/status')

# Process PDF function
def process_pdf(pdf_path):
    send_status("Processing PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    documents = text_splitter.split_documents(pages)
    return [doc.page_content.replace('\n', ' ') for doc in documents]

# Process web links function
def process_web_links(links):
    send_status("Processing web links...")
    loader = UnstructuredURLLoader(urls=links)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    documents = text_splitter.split_documents(documents)
    return [doc.page_content.replace('\n', ' ') for doc in documents]

# Document processing function
def process_documents(document_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    return text_splitter.split_documents([Document(page_content=text) for text in document_texts])

# Prepare vectors for FAISS
def prepare_vectors(documents):
    document_texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(document_texts)
    return document_texts, np.array(vectors)

# Create FAISS index function
def create_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

# Rephrase question with history
def rephrase_question_with_history(chat_history, question):
    template = """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
    
    <chat_history>
      {chat_history}
    </chat_history>
    
    Follow Up Input: {question}
    Standalone question:
    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(chat_history=chat_history, question=question)
    response = LLM(formatted_prompt)
    return response.content if hasattr(response, 'content') else str(response)

# Generate answer function
def generate_answer(query, combined_texts, index, chat_history):
    standalone_question = rephrase_question_with_history(chat_history, query)
    print(f'standalone question updated with history inside generate_answer function')
    query_vector = embeddings.embed_query(standalone_question)
    print(f'query vector generated')
    query_vector = np.array(query_vector).reshape((1, -1))
    positions = index.search(query_vector, k=4)[1]
    print(f'index positions in vector space found')
    final_context = ' '.join([combined_texts[pos] for pos in positions[0]])
    print(f'final context generated')
    template = """
    You are an expert researcher. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    If the question is not related to the context or chat history, politely respond that you are tuned to only answer questions that are related to the context.
    
    <context>
      {context}
    </context>
    
    <chat_history>
      {chat_history}
    </chat_history>
    
    Question: {question}
    Helpful answer in markdown:
    """
    prompt = PromptTemplate.from_template(template)
    print(f'Initial prompt initialized')
    formatted_prompt = prompt.format(context=final_context, chat_history=chat_history, question=standalone_question)
    print(f'final prompt generated')
    response = LLM(formatted_prompt)
    print(f'Answer also generated, Good')
    return response.content if hasattr(response, 'content') else str(response)

# Route for creating a new chat session
@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    session['session_id'] = session_id  # Store the session ID in the session
    session['chat_history'] = []  # Initialize an empty chat history for the new session
    return redirect(url_for('chatbot_index', session_id=session_id))

# Route for deleting the current chat session
@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    session.pop('chat_history', None)  # Remove chat history from the session
    session.pop('session_id', None)  # Remove session ID from the session
    return redirect(url_for('chatbot_index'))

# Chatbot index route
@app.route('/chatbot', methods=['GET'])
def chatbot_index():
    return render_template('chatbot_index.html')

# Upload and process request route
@app.route('/chatbot/upload', methods=['POST'])
def chatbot_upload():
    pdf_file = request.files.get('pdf_file')
    links = request.form.get('links')
    question = request.form.get('question')
    history = request.form.get('history', '')

    if not question:
        return jsonify({'error': 'Please provide a question.'}), 400

    if not pdf_file and not links:
        return jsonify({'error': 'Please upload a PDF or provide links.'}), 400

    pdf_path = None
    if pdf_file:
        upload_dir = 'uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        pdf_path = os.path.join(upload_dir, pdf_file.filename)
        pdf_file.save(pdf_path)

    web_links = [link.strip() for link in links.split(',')] if links else []

    try:
        combined_texts = []
        if pdf_file:
            combined_texts.extend(process_documents(process_pdf(pdf_path)))
        if web_links:
            combined_texts.extend(process_documents(process_web_links(web_links)))
        print(f'combined texts generated from pdf as well as link')
        combined_texts, combined_vectors = prepare_vectors(combined_texts)
        faiss_index = create_faiss_index(combined_vectors)
        print(f'faiss index step completed')
        if 'chat_history' not in session:
            session['chat_history'] = []
        print(f'chat history step completed')
        session['chat_history'].append(f"Human: {question}")
        print(f'chat history appended to session with question')
        
        answer = generate_answer(question, combined_texts, faiss_index, '\n'.join(session['chat_history']))
        print(f'Answer generated')
        session['chat_history'].append(f"Assistant: {answer}")
        print(f'chat history appended to session with answer')
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # socketio.run(app, debug=False)
    app.run(debug=True, threaded=False)