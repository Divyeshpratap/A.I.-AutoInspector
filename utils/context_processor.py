# utils/context_processors.py

from models.database import User
from flask import session

def inject_user():
    """
    Injects the current user into the template context.
    """
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        return {'user': user}
    return {'user': None}
