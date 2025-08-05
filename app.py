from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from datetime import datetime
import json

app = Flask(__name__)
CORS(app, origins=['*'])

# Simple storage
jobs = {}
api_keys = {
    "test-key-123": {
        "email": "test@example.com",
        "tier": "free",
        "credits": 3
    }
}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    try:
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if api_key not in api_keys:
            return jsonify({'error': 'Invalid API key'}), 401
        
        key_data = api_keys[api_key]
        if key_data['tier'] == 'free' and key_data['credits'] <= 0:
            return jsonify({'error': 'No credits remaining'}), 402
        
        data = request.json
        job_id = str(uuid.uuid4())
        
        jobs[job_id] = {
            'status': 'processing',
            'created_at': datetime.now().isoformat()
        }
        
        if key_data['tier'] == 'free':
            api_keys[api_key]['credits'] -= 1
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Analysis started',
            'credits_remaining': api_keys[api_key]['credits']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def check_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)