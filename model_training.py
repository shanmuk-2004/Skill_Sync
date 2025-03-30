import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import os

# Create the 'model' directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the dataset from Excel
df = pd.read_csv('data/job_dataset.csv')  # Update the file path if needed

# Define the list of IT job roles to recommend
it_job_roles = [
    "Software Engineer", "Software Developer", "Full Stack Developer", "Frontend Developer", "Backend Developer",
    "Web Developer", "Mobile App Developer", "Game Developer", "Embedded Software Engineer", "API Developer",
    "Systems Software Engineer", "DevOps Engineer", "Site Reliability Engineer (SRE)", "Test Automation Engineer",
    "Application Developer", "Firmware Developer", "Blockchain Developer", "Low-Code/No-Code Developer",
    "Cybersecurity Analyst", "Cybersecurity Engineer", "Information Security Analyst", "Ethical Hacker / Penetration Tester",
    "Security Architect", "Security Engineer", "Security Consultant", "SOC Analyst (Security Operations Center)",
    "Malware Analyst", "Incident Response Analyst", "Cryptographer", "Digital Forensics Analyst", "Application Security Engineer",
    "Cloud Security Engineer", "Data Scientist", "Data Analyst", "Data Engineer", "Big Data Engineer", "Machine Learning Engineer",
    "AI Engineer", "Business Intelligence Analyst", "BI Developer", "Deep Learning Engineer", "NLP Engineer (Natural Language Processing)",
    "Computer Vision Engineer", "Statistician", "Data Architect", "Quantitative Analyst", "Cloud Engineer", "Cloud Architect",
    "AWS Solutions Architect", "Azure Cloud Engineer", "Google Cloud Engineer", "Kubernetes Administrator", "Cloud Consultant",
    "Multi-Cloud Engineer", "Network Engineer", "Network Administrator", "System Administrator", "IT Support Specialist",
    "Help Desk Technician", "IT Technician", "IT Consultant", "VoIP Engineer", "Systems Engineer", "Infrastructure Engineer",
    "Database Administrator (DBA)", "Storage Administrator", "Linux Administrator", "Windows Administrator", "Technical Support Engineer",
    "Wireless Network Engineer", "AI Engineer", "Machine Learning Engineer", "Deep Learning Engineer", "NLP Engineer",
    "Computer Vision Engineer", "Reinforcement Learning Engineer", "AI Research Scientist", "AI Consultant", "UI/UX Designer",
    "UX Researcher", "Product Designer", "Human-Computer Interaction (HCI) Specialist", "Interaction Designer", "Visual Designer",
    "QA Engineer", "Manual Tester", "Automation Tester", "Performance Tester", "Penetration Tester", "Test Engineer",
    "Software Quality Assurance (SQA) Engineer", "IT Manager", "IT Director", "Chief Information Officer (CIO)", "Chief Technology Officer (CTO)",
    "Technical Lead", "Product Manager", "Program Manager", "IT Project Manager", "Engineering Manager", "Scrum Master", "Agile Coach",
    "ERP Consultant", "SAP Consultant", "Salesforce Developer", "Microsoft Dynamics Consultant", "Database Administrator (DBA)",
    "Data Architect", "Database Developer", "NoSQL Developer", "SQL Developer", "ETL Developer", "Blockchain Developer",
    "Blockchain Architect", "IoT Engineer", "Quantum Computing Engineer", "Robotics Engineer", "IT Auditor", "IT Trainer",
    "Digital Transformation Consultant", "IT Compliance Manager", "IT Risk Analyst", "IT Procurement Specialist", "IT Governance Specialist",
    "IT Legal Consultant"
]

# Filter the dataset to include only the IT job roles
df = df[df['Role'].isin(it_job_roles)]

# Label Encoding for categorical columns
# Label Encoding for categorical columns
label_encoders = {}
categorical_columns = ['Experience', 'Qualifications', 'Work Type', 'Preference', 'Job Title', 'Role', 'Company']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # Add 'Unknown' to the classes
    le.classes_ = np.append(le.classes_, 'Unknown')
    label_encoders[col] = le

# Text Vectorization for 'Job Description' and 'Skills'
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
job_desc_tfidf = tfidf.fit_transform(df['Job Description'])
skills_tfidf = tfidf.fit_transform(df['Skills'])

# Dimensionality Reduction using Truncated SVD
svd = TruncatedSVD(n_components=100)
job_desc_svd = svd.fit_transform(job_desc_tfidf)
skills_svd = svd.fit_transform(skills_tfidf)

# Combine all features
X = np.hstack((df[categorical_columns].values, job_desc_svd, skills_svd))
y = df['Role']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Feedforward Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, batch_size=1000, verbose=True, alpha=0.01)
mlp.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and preprocessing objects
joblib.dump(mlp, 'model/mlp_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
joblib.dump(svd, 'model/svd.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Model training and saving completed!")