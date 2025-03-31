from flask import Flask, send_file, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

APP_ID = "1b4bc52c"
APP_KEY = "3934da55d5634309d3d364a3fc69724c"

@app.route('/')
def home():
    return send_file('job_search.html')  # ← Updated here

@app.route('/job_search')
def job_search_page():
    return render_template('job_search.html')

@app.route('/search_jobs', methods=['GET'])
def search_jobs():
    role = request.args.get('role', 'Software Engineer')
    location = request.args.get('location', 'India')
    country = 'in'

    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {
        'app_id': APP_ID,
        'app_key': APP_KEY,
        'what': role,
        'where': location,
        'results_per_page': 10,
        'content-type': 'application/json'
    }

    response = requests.get(url, params=params)

    # Debugging: Print the response to see if it's valid
    print(f"API Response Status: {response.status_code}")
    print(f"API Response Body: {response.text}")

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({'error': 'Failed to fetch jobs', 'status': response.status_code, 'message': response.text}), response.status_code

