# app.py

import os
import re
import json  # Added for safe JSON parsing
import uuid
import logging
import requests
import base64
import statistics
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, flash, redirect, url_for,
    session, jsonify, send_file, current_app
)
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image
from pytz import timezone
from bs4 import BeautifulSoup
import numpy as np
import torch
import matplotlib.pyplot as plt

from mmdet.apis import init_detector, inference_detector
from base64 import b64encode
from matplotlib.patches import Rectangle

from models.database import db, User, Agent, Customer, Order, InferenceResult
from models.cv.cv_model import (
    model, det_model, transform, get_best_vehicle_box, adjust_bbox,
    process_damage_analysis, calculate_repair_cost_analysis, visualize_damage_and_costs
)
from models.nlp.nlp_model import (
    LLM, embeddings, rephrase_question_with_history, process_pdf,
    process_web_links, process_documents, prepare_vectors, create_faiss_index,
    generate_answer
)
from utils.geolocation import calculate_distance, find_nearest_agent, find_next_nearest_agent, reverse_geocode
from utils.web_scraper import (
    fetch_search_results, fetch_webpage_content, extract_price_from_webpage, extract_prices_from_snippets
)
from utils.data_helpers import convert_to_serializable

from utils.helpers import basename_filter
from utils.filters import to_user_timezone
from utils.context_processor import inject_user
from utils.sqlalchemy_events import register_listeners

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize Flask app with absolute instance_path
instance_dir = (Path(__file__).parent / 'instance').resolve()
app = Flask(__name__, instance_path=str(instance_dir))

# Ensure the instance folder exists
instance_dir.mkdir(parents=True, exist_ok=True)

# Set the database URI using absolute path
db_path = instance_dir / 'site.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path.as_posix()}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Set Flask secret keys
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_super_secret_key')  # Loaded from .env
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default_super_secret_key')  # Loaded from .env

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, async_mode='threading')


# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Retrieve API keys from environment
gmaps_api_key = os.getenv('GoogleMaps_API_KEY')
gsearch_api_key = os.getenv('GoogleSearch_API_KEY')
gsearch_engine_id = os.getenv('GoogleSearch_engine_id')

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create all database tables
with app.app_context():
    db.create_all()

# Register template filters
app.jinja_env.filters['basename'] = basename_filter
app.jinja_env.filters['to_user_timezone'] = to_user_timezone

# Register context processors
app.context_processor(inject_user)

# Register SQLAlchemy event listeners
register_listeners()

# Define WebSocket status message function
def send_status(message):
    socketio.emit('status', message, namespace='/status')

# --------------------
# Route Definitions
# --------------------

if not os.path.exists('static/results'):
    os.makedirs('static/results')
    print("Created 'results' directory.")


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
        # Update the password only if a new one is provided
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
            flash('Login Unsuccessful. Please check username and password', 'error')
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
        if customer:
            customer.mobile_number = request.form['mobile_number']
        else:
            # If customer entry doesn't exist, create one
            customer = Customer(user_id=user.id, mobile_number=request.form['mobile_number'])
            db.session.add(customer)
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
            .outerjoin(InferenceResult, Order.prediction_id == InferenceResult.id)
            .filter(Order.user_id == user.id)
            .all()
        )

        # Prepare data to pass to the template
        inference_data = []
        for order, agent, inference_result, agent_username in inferences:
            if inference_result and isinstance(inference_result.inference_details, dict):
                inference_details = inference_result.inference_details
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
            inference_result = None
            if order.prediction_id is not None:
                inference_result = InferenceResult.query.filter_by(id=order.prediction_id).first()
            
            hours_since_created = None
            if order.created_at and order.timezone:
                order_tz = timezone(order.timezone)
                now_in_order_tz = datetime.now(order_tz)
                created_at_tz_aware = order.created_at.replace(tzinfo=timezone('UTC')).astimezone(order_tz)
                time_diff = now_in_order_tz - created_at_tz_aware
                hours_since_created = time_diff.total_seconds() // 3600

            if inference_result and isinstance(inference_result.inference_details, dict):
                inference_details = inference_result.inference_details
            else:
                inference_details = {}

            image_path = inference_details.get('inf_image', '')
            car_model = inference_details.get('car_model')
            car_year = inference_details.get('car_year')

            orders_data.append({
                'id': order.id,
                'status': order.status,
                'image_path': image_path,
                'hours_since_created': hours_since_created
            })
        
        return render_template(
            'agent_dashboard.html',
            orders=orders_data,
            agent=agent,
            car_model=f"{car_model} {car_year}" if car_model and car_year else ''
        )
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
        flash("You need to log in as an agent to update your location.", "warning")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    agent = Agent.query.filter_by(user_id=user_id).first()
    
    if agent:
        agent.latitude = request.form['latitude']
        agent.longitude = request.form['longitude']
        db.session.commit()
        flash('Location updated successfully!', 'success')
    else:
        new_agent = Agent(
            user_id=user_id,
            latitude=request.form['latitude'],
            longitude=request.form['longitude']
        )
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
        inference_details = inference_result.inference_details
    else:
        inference_details = {}
    
    original_path = inference_details.get('orig_image', '')
    prediction_path = inference_details.get('inf_image', '')
    car_model = inference_details.get('car_model')
    car_year = inference_details.get('car_year')
    damage_details = inference_details.get('damage_info')
    customer = Customer.query.filter_by(user_id=order.user_id).first()
    customer_phone = customer.mobile_number if customer else 'Customer phone number not available, please contact via email'
    
    user = User.query.get(customer.user_id) if customer else None
    customer_email = user.email if user else 'Customer email not available'

    return render_template(
        'order_details.html',
        google_maps_api_key=gmaps_api_key,
        order=order,
        agent=agent,
        customer_phone=customer_phone,
        customer_email=customer_email,
        original_path=original_path,
        prediction_path=prediction_path,
        car_model=f"{car_model} {car_year}" if car_model and car_year else '',
        damage_details=damage_details
    )

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
        chosen_damages=chosen_damages,
        repair_cost=repair_cost
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

# --------------------
# Car Price Fetching Routes
# --------------------

def get_car_price_from_web(api_key, search_engine_id, query, num_results=4):
    search_results = fetch_search_results(api_key, search_engine_id, query, num_results)
    all_prices = []
    for result in search_results:
        url = result.get('link')
        if url:
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
    query = f'{car_year} {car_model} MSRP'
    avg_price = get_car_price_from_web(gsearch_api_key, gsearch_engine_id, query, num_results=2)
    
    if avg_price:
        session['car_price'] = f'{avg_price:.2f}'
        car_price = f'{avg_price:.2f}'
    else:
        car_price = '0.00'  # Default to '0.00' if the price is not found
    
    return render_template(
        'upload.html',
        car_price=car_price,
        car_model=car_model,
        car_year=car_year,
        step='price_fetched',
        google_maps_api_key=gmaps_api_key
    )

@app.route('/check-car-damages', methods=['GET', 'POST'])
def check_car_damages():
    if request.method == 'GET':
        # Fetch the values from session and render the page
        car_price = request.args.get('car_price', '')
        car_year = request.args.get('car_year', '')
        car_model = request.args.get('car_model', '')
        return render_template(
            'upload.html',
            car_price=car_price,
            car_year=car_year,
            car_model=car_model,
            google_maps_api_key=gmaps_api_key
        )
    
    elif request.method == 'POST':
        car_price = request.form.get('car_price')
        car_year = request.form.get('car_year')
        car_model = request.form.get('car_model')
        file = request.files.get('file')

        from werkzeug.utils import secure_filename  # Added for secure file handling

        ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

        if file and allowed_file(file.filename):
            try:
                image = Image.open(file.stream)
                img_tensor = transform(image).unsqueeze(0).to(device)
                det_output = det_model(img_tensor)[0]

                best_bbox = get_best_vehicle_box(det_output)
                if best_bbox is not None:
                    bbox_np = best_bbox.detach().cpu().numpy().astype(int)
                    img_width, img_height = image.size
                    adjusted_bbox = adjust_bbox(bbox_np, img_width, img_height, margin=0.1)

                    # Crop the image to the adjusted bounding box
                    cropped_img = image.crop(adjusted_bbox)

                    # Save the original and cropped images
                    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
                    orig_filename = f'{timestamp}_original.jpg'
                    cropped_filename = f'{timestamp}_cropped.jpg'
                    orig_filename_secure = secure_filename(orig_filename)
                    cropped_filename_secure = secure_filename(cropped_filename)
                    orig_filepath = os.path.join('results', orig_filename_secure).replace('\\', '/')
                    cropped_filepath = os.path.join('results', cropped_filename_secure).replace('\\', '/')

                    image.save(os.path.join(app.static_folder, orig_filepath))
                    Image.fromarray(np.array(cropped_img)).save(os.path.join(app.static_folder, cropped_filepath))

                    return render_template(
                        'upload.html',
                        orig_image=orig_filepath,
                        cropped_image=cropped_filepath,
                        step='confirm',
                        car_price=car_price,
                        car_year=car_year,
                        car_model=car_model,
                        google_maps_api_key=gmaps_api_key
                    )
                else:
                    flash("No vehicle detected in the image. Please upload a clearer image.", 'error')
                    return redirect(url_for('check_car_damages'))

            except Exception as e:
                logger.error(f"Error processing image in check_car_damages: {e}")
                flash("Error processing image.", 'error')
                return redirect(url_for('check_car_damages'))
        else:
            flash("Unsupported file type or no file found", 'error')
            return redirect(url_for('check_car_damages'))

# --------------------
# Damage Inference Route
# --------------------

@app.route('/run-damage-inference', methods=['POST'])
def run_damage_inference():
    try:
        # Retrieve form data
        cropped_image_path = request.form.get('cropped_image_path')
        orig_image_path = request.form.get('orig_image_path')
        car_price = request.form.get('car_price')
        car_year = request.form.get('car_year')
        car_model = request.form.get('car_model')
        
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
        classification_dict = {
            1: 'dent',
            2: 'scratch',
            3: 'crack',
            4: 'glass_shatter',
            5: 'lamp_broken',
            6: 'tire_flat'
        }
        
        # Process damage results
        damage_info, damage_masks = process_damage_analysis(damage_result, classification_dict)
        
        # Define damage factors
        damage_factors = {
            'dent': 0.3,
            'scratch': 0.2,
            'crack': 0.3,
            'glass_shatter': 0.35,
            'lamp_broken': 0.4,
            'tire_flat': 0.1
        }
        
        # Calculate repair cost
        image_size = cropped_image.size
        repair_cost, chosen_damages = calculate_repair_cost_analysis(
            damage_info, car_price, car_year, damage_factors, image_size
        )
        PALETTE = [
            (255, 182, 193), (0, 168, 225), (0, 255, 0),
            (128, 0, 128), (255, 255, 0), (227, 0, 57)
        ]
        final_cropped_result = model.show_result(
            cropped_img_np,
            damage_result,
            show=False,
            score_thr=0.5,
            text_color=PALETTE,
            mask_color=PALETTE,
            font_size=20
        )
        
        # Generate and save inference result image
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        inf_filename = f'{timestamp}_inference.jpg'
        inf_filename_secure = secure_filename(inf_filename)
        inf_filepath = os.path.join('results', inf_filename_secure).replace('\\', '/')
        Image.fromarray(final_cropped_result).save(os.path.join(app.static_folder, inf_filepath))
        
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

        inference_details_serializable = {
            key: convert_to_serializable(value)
            for key, value in inference_details.items()
        }
        
        inference_result = InferenceResult(
            user_id=session.get('user_id', 999),  # Use actual user_id from session or default to 999
            image_path=inf_filepath,
            original_path=orig_image_path,
            inference_details=inference_details_serializable
        )
        
        # Add and commit the inference result to the database
        db.session.add(inference_result)
        db.session.commit()
        inference_id = inference_result.id
        session['inference_id'] = inference_id

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
        logger.error(f"Error processing damage inference in run_damage_inference: {e}")
        flash("Error processing damage inference.", 'error')
        return redirect(url_for('check_car_damages'))

# --------------------
# Detailed Analysis Route
# --------------------

@app.route('/detailed-analysis', methods=['POST'])
def detailed_analysis():
    try:
        # Retrieve data passed from the previous step
        repair_cost = float(request.form.get('repair_cost'))
        damage_info = json.loads(request.form.get('damage_info', '{}'))  # Safe conversion
        chosen_damages = json.loads(request.form.get('chosen_damages', '[]'))  # Safe conversion

        # Generate visualizations
        img = visualize_damage_and_costs(damage_info, repair_cost, chosen_damages)

        # Convert plot to PNG image and base64 encode it
        buf = BytesIO()
        img.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        
        return render_template(
            'detailed_analysis.html',
            repair_cost=repair_cost,
            image_base64=image_base64
        )

    except Exception as e:
        logger.error(f"Error generating detailed analysis: {e}")
        flash("Error generating detailed analysis.", 'error')
        return redirect(url_for('check_car_damages'))

# --------------------
# Repair Request Route
# --------------------

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
    
    address = reverse_geocode(user_latitude, user_longitude, gmaps_api_key)
    nearest_agent, min_distance = find_nearest_agent(user_latitude, user_longitude)
    max_distance = 500  # Define maximum distance in kilometers or miles as per your requirement
    if nearest_agent is None or min_distance > max_distance:
        flash("No nearby agents found within the maximum distance.", "error")
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

# --------------------
# Chatbot Routes
# --------------------

@app.route('/new_chat', methods=['POST'])
def new_chat():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    session['session_id'] = session_id  # Store the session ID in the session
    session['chat_history'] = []  # Initialize an empty chat history for the new session
    return redirect(url_for('chatbot_index', session_id=session_id))

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    session.pop('chat_history', None)  # Remove chat history from the session
    session.pop('session_id', None)  # Remove session ID from the session
    return redirect(url_for('chatbot_index'))

@app.route('/chatbot', methods=['GET'])
def chatbot_index():
    return render_template('chatbot_index.html')

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
        upload_dir = Path('uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename_secure = secure_filename(pdf_file.filename)
        pdf_path = upload_dir / filename_secure
        pdf_file.save(pdf_path)

    web_links = [link.strip() for link in links.split(',')] if links else []

    try:
        combined_texts = []
        if pdf_file:
            combined_texts.extend(process_documents(process_pdf(str(pdf_path))))
        if web_links:
            combined_texts.extend(process_documents(process_web_links(web_links)))

        combined_texts, combined_vectors = prepare_vectors(combined_texts)
        faiss_index = create_faiss_index(combined_vectors)

        if 'chat_history' not in session:
            session['chat_history'] = []

        session['chat_history'].append(f"Human: {question}")
        
        answer = generate_answer(
            question,
            combined_texts,
            faiss_index,
            '\n'.join(session['chat_history'])
        )
        session['chat_history'].append(f"Assistant: {answer}")
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500

# --------------------
# Main Execution
# --------------------

if __name__ == '__main__':
    # Uncomment the following line if you want to use SocketIO
    # socketio.run(app, debug=False)
    app.run(debug=True, threaded=False)
