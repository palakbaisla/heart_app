import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Function to train and save models
def train_and_save_models():
    # Load dataset
    df = pd.read_csv("heart.csv")
    
    # Preprocess data (label encoding)
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Features & target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    
    with open('log_model.pkl', 'wb') as f:
        pickle.dump(log_model, f)

    # Train Decision Tree
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    
    with open('tree_model.pkl', 'wb') as f:
        pickle.dump(tree_model, f)

    # Evaluate accuracy
    y_pred_log = log_model.predict(X_test)
    y_pred_tree = tree_model.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log)}")
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree)}")

# Uncomment to train and save models (run once)
# train_and_save_models()

# Streamlit App Code
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction App")

st.markdown("""
Welcome! This app predicts the **likelihood of heart disease** based on user input.  
Choose a model below, enter the patient's data, and click **Predict**!
""")

# Load models
with open('log_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open('tree_model.pkl', 'rb') as f:
    tree_model = pickle.load(f)

# Model selection
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree"])

# Input fields (same as before)
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak (ST depression)", step=0.1, value=1.0)
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Encode input values
sex = 1 if sex == "Male" else 0
cp = {"TA": 3, "ATA": 2, "NAP": 1, "ASY": 0}[cp]
fbs = 1 if fbs == "Yes" else 0
restecg = {"Normal": 1, "ST": 2, "LVH": 0}[restecg]
exang = 1 if exang == "Yes" else 0
slope = {"Up": 2, "Flat": 1, "Down": 0}[slope]

# Define the correct column names for input data
input_columns = ['Age','Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

# Prepare input for prediction with correct column names
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]],
                          columns=input_columns)

# Prediction
if st.button("Predict"):
    model = log_model if model_choice == "Logistic Regression" else tree_model
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")

# Function to make predictions (for accuracy check)
def make_prediction(model, X_data):
    return model.predict(X_data)

# Option to show accuracy (based on test data)
if st.checkbox("Show Model Accuracy (based on test data)"):
    model = log_model if model_choice == "Logistic Regression" else tree_model
    
    # Re-load test data (same preprocessing applied)
    df = pd.read_csv("heart.csv")
    le = LabelEncoder()
    for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
        df[col] = le.fit_transform(df[col])

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = make_prediction(model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
# Footer
st.markdown("""
---
### Made with ❤️ by [Palak]
""")                    
