# utils/geolocation.py
import os
import requests
import logging
from math import radians, sin, cos, sqrt, atan2
from models.database import Agent

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two geographical points using the Haversine formula.
    Args:
    - lat1, lon1: Latitude and Longitude of the first point.
    - lat2, lon2: Latitude and Longitude of the second point.
    
    Returns:
    - Distance in kilometers.
    """
    R = 6371.0  # Radius of the Earth in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def find_nearest_agent(user_latitude, user_longitude):
    """
    Find the nearest agent based on the user's location.
    Args:
    - user_latitude: Latitude of the user.
    - user_longitude: Longitude of the user.
    
    Returns:
    - Nearest agent and distance in kilometers.
    """
    nearest_distance = float('inf')
    nearest_agent = None
    agents = Agent.query.all()
    for agent in agents:
        if agent.latitude is not None and agent.longitude is not None:
            distance = calculate_distance(user_latitude, user_longitude, agent.latitude, agent.longitude)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_agent = agent
    return nearest_agent, nearest_distance


def find_next_nearest_agent(lat, lon, exclude_agent_id):
    """
    Find the next nearest agent, excluding a specific agent.
    Args:
    - lat, lon: Latitude and Longitude of the user.
    - exclude_agent_id: ID of the agent to exclude.
    
    Returns:
    - Next nearest agent.
    """
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


def reverse_geocode(lat, lng, api_key):
    """
    Converts latitude and longitude to a formatted address using Google Geocoding API.
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        results = response.json().get('results', [])
        if results:
            return results[0].get('formatted_address')
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in reverse_geocode: {e}")
    return None