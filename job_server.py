from flask import Flask, render_template, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

APP_ID = "1b4bc52c"
APP_KEY = "3934da55d5634309d3d364a3fc69724c"

@app.route('/')
def home():
    return render_template('job_search.html')  # ← Updated here

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
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5051, debug=True)
