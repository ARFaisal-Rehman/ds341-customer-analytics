from flask import Flask, render_template, jsonify, send_from_directory
import os
import json
import pandas as pd
import socket

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_PATH = os.path.join(BASE_DIR, 'src', 'processed_data.csv')

@app.route('/')
def index():
    return render_template('overview.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/segments')
def segments():
    return render_template('segments.html')

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/api/metrics')
def get_metrics():
    try:
        with open(os.path.join(RESULTS_DIR, 'model_metrics.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({})

@app.route('/api/clusters')
def get_clusters():
    try:
        with open(os.path.join(RESULTS_DIR, 'cluster_summary.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify([])

@app.route('/api/rules')
def get_rules():
    try:
        with open(os.path.join(RESULTS_DIR, 'association_rules.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify([])

@app.route('/api/stats')
def get_stats():
    try:
        df = pd.read_csv(DATA_PATH)
        churn_counts = df['Churn'].value_counts().to_dict()
        total_customers = len(df)
        churn_rate = (churn_counts.get(1, 0) / total_customers) * 100
        
        return jsonify({
            'total_customers': total_customers,
            'churn_rate': round(churn_rate, 2),
            'churn_counts': churn_counts or {
                'No': 0,
                'Yes': 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/results/<path:filename>')
def custom_static(filename):
    return send_from_directory(RESULTS_DIR, filename)

def get_free_port(start_port=5000):
    """Finds a free port starting from start_port."""
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Set SO_REUSEADDR to avoid waiting for timeout states
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', port))
                return port
        except OSError:
            port += 1

if __name__ == '__main__':
    port = get_free_port(5000)
    print(f"Starting server on port {port}...")
    app.run(debug=True, port=port)
