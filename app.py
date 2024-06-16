import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import joblib
import os
# load data
df = pd.read_csv("data.csv")
st.write(df.head())
# Function to load the model
def load_model(filename):
    if not os.path.isfile(filename):
        st.error(f"Error: The file '{filename}' does not exist.")
        return None
    return joblib.load(filename)

st.title('Breast Cancer Prediction App')

# Default values for a benign tumor
default_values = {
    'radius_mean': 12.0,
    'texture_mean': 14.0,
    'perimeter_mean': 78.0,
    'area_mean': 450.0,
    'smoothness_mean': 0.09,
    'compactness_mean': 0.05,
    'concavity_mean': 0.03,
    'concave_points_mean': 0.02,
    'symmetry_mean': 0.18,
    'fractal_dimension_mean': 0.06,
    'radius_se': 0.3,
    'texture_se': 1.0,
    'perimeter_se': 2.0,
    'area_se': 20.0,
    'smoothness_se': 0.006,
    'compactness_se': 0.02,
    'concavity_se': 0.02,
    'concave_points_se': 0.01,
    'symmetry_se': 0.03,
    'fractal_dimension_se': 0.006,
    'radius_worst': 14.0,
    'texture_worst': 16.0,
    'perimeter_worst': 90.0,
    'area_worst': 500.0,
    'smoothness_worst': 0.11,
    'compactness_worst': 0.12,
    'concavity_worst': 0.1,
    'concave_points_worst': 0.08,
    'symmetry_worst': 0.25,
    'fractal_dimension_worst': 0.08
}

# Input fields for the features
input_features = {}
for feature, default in default_values.items():
    input_features[feature] = st.number_input(feature.replace('_', ' ').capitalize(), min_value=0.0, value=default)

# Load the model with the corrected file path
model_path = 'breast2_cancer_model(1).pkl'
model = load_model(model_path)

if model and st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame([input_features])
    
    # Make prediction
    prediction = model.predict(input_data)
    st.write(f'The prediction is: {"Malignant" if prediction[0] == 1 else "Benign"}')
