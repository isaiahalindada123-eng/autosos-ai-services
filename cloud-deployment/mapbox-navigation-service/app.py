 from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Mapbox secret token for navigation (server-side only)
MAPBOX_SECRET_TOKEN = os.getenv('MAPBOX_SECRET_TOKEN', 'sk.eyJ1IjoiYXV0b3NvczEyMyIsImEiOiJjbWdvNTVkMXQweXlrMmpyNTRuMmtqcHN2In0.GcRVIJQCERTgm_AOizdffgt')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'mapbox-navigation-service',
        'version': '1.0.0'
    })

@app.route('/api/directions', methods=['POST'])
def get_directions():
    """
    Get directions between two points using Mapbox Directions API
    This endpoint uses the secret token for navigation
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        origin_lat = data.get('origin_lat')
        origin_lng = data.get('origin_lng')
        destination_lat = data.get('destination_lat')
        destination_lng = data.get('destination_lng')
        
        if not all([origin_lat, origin_lng, destination_lat, destination_lng]):
            return jsonify({
                'error': 'Missing required parameters: origin_lat, origin_lng, destination_lat, destination_lng'
            }), 400
        
        # Build Mapbox Directions API URL
        url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{origin_lng},{origin_lat};{destination_lng},{destination_lat}"
        
        params = {
            'access_token': MAPBOX_SECRET_TOKEN,
            'geometries': 'polyline',
            'overview': 'full',
            'steps': 'true',
            'annotations': 'duration,distance'
        }
        
        # Make request to Mapbox API
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': f'Mapbox API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/geocoding', methods=['POST'])
def geocoding():
    """
    Geocoding service using Mapbox Geocoding API
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        query = data.get('query')
        lat = data.get('lat')
        lng = data.get('lng')
        
        if not query:
            return jsonify({'error': 'Missing required parameter: query'}), 400
        
        # Build Mapbox Geocoding API URL
        if lat and lng:
            # Reverse geocoding
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lng},{lat}.json"
        else:
            # Forward geocoding
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{query}.json"
        
        params = {
            'access_token': MAPBOX_SECRET_TOKEN,
            'country': 'PH',
            'types': 'address,poi,place',
            'limit': 5
        }
        
        # Make request to Mapbox API
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': f'Mapbox API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/matrix', methods=['POST'])
def distance_matrix():
    """
    Distance matrix service using Mapbox Matrix API
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        coordinates = data.get('coordinates')
        
        if not coordinates or len(coordinates) < 2:
            return jsonify({'error': 'At least 2 coordinates required'}), 400
        
        # Format coordinates for Mapbox Matrix API
        coords_str = ';'.join([f"{coord['lng']},{coord['lat']}" for coord in coordinates])
        
        # Build Mapbox Matrix API URL
        url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coords_str}"
        
        params = {
            'access_token': MAPBOX_SECRET_TOKEN,
            'annotations': 'duration,distance'
        }
        
        # Make request to Mapbox API
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                'error': f'Mapbox API error: {response.status_code}',
                'details': response.text
            }), response.status_code
            
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
