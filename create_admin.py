import argparse
from app import app, db, User
from werkzeug.security import generate_password_hash
import os

if not os.path.exists('instance'):
    os.makedirs('instance')
    print("Created 'instance' directory.")

def create_admin(username, email, password):
    with app.app_context():
        admin_password = generate_password_hash(password)
        admin_user = User(username=username, email=email, password=admin_password, role='admin')
        db.session.add(admin_user)
        db.session.commit()
        print(f"Admin account '{username}' created successfully.")

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Create an admin user for the application.")
    
    # Adding arguments with default values
    parser.add_argument('--username', type=str, default='administrator', help="Admin username (default: 'administrator')")
    parser.add_argument('--email', type=str, default='admin@xyz.com', help="Admin email (default: 'admin@xyz.com')")
    parser.add_argument('--password', type=str, default='admin', help="Admin password (default: 'admin')")

    # Parse arguments
    args = parser.parse_args()

    # Call the create_admin function with arguments
    create_admin(args.username, args.email, args.password)
