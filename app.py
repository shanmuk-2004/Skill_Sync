from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model and preprocessing objects
mlp = joblib.load('mlp_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
svd = joblib.load('svd.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('Job_Recommendation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from the form
    user_input = {
        'Experience': request.form['experience'],
        'Qualifications': request.form.getlist('education[]'),
        'Work Type': request.form['work_type'],
        'Preference': request.form['gender'],
        'Job Description': ' '.join(request.form.getlist('skills[]')),
        'Skills': ' '.join(request.form.getlist('skills[]'))
    }

    # Preprocess user input
    user_data_encoded = []
    for col in label_encoders:
        le = label_encoders[col]
        if col in user_input:
            if isinstance(user_input[col], list):  # Handle multi-value fields
                value = user_input[col][0]  # Use the first value
            else:
                value = user_input[col]

            # Handle unseen labels
            if value not in le.classes_:
                value = 'Unknown'  # Map unseen values to 'Unknown'

            user_data_encoded.append(le.transform([value])[0])
        else:
            user_data_encoded.append(0)  # Default value if not provided

    # Vectorize text data
    job_desc_tfidf = tfidf.transform([user_input['Job Description']])
    skills_tfidf = tfidf.transform([user_input['Skills']])

    # Apply SVD
    job_desc_svd = svd.transform(job_desc_tfidf)
    skills_svd = svd.transform(skills_tfidf)

    # Combine all features
    user_data_encoded = np.array(user_data_encoded).reshape(1, -1)  # Reshape to (1, n)
    user_features = np.hstack((user_data_encoded, job_desc_svd, skills_svd))
    user_features = scaler.transform(user_features)

    # Get probabilities for all job roles
    probabilities = mlp.predict_proba(user_features)[0]  # Get probabilities for the first sample

    # Get the top N job roles with the highest probabilities
    top_n = 5  # Number of recommendations to return
    top_indices = np.argsort(probabilities)[-top_n:][::-1]  # Indices of top N job roles
    role_encoder = label_encoders['Role']

    # Create a list of recommended job roles with their probabilities
    recommendations = []
    for index in top_indices:
        role = role_encoder.inverse_transform([index])[0]  # Convert index to job role name
        probability = probabilities[index]  # Probability of the job role
        recommendations.append({
            'Role': role,
            'Probability': float(probability)  # Convert numpy float to Python float
        })

    # Return the recommendations
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
