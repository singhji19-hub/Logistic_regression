import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="Titanic Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")

# --- 2. LOAD & PREPROCESS (Your Notebook Logic) ---
@st.cache_data # This makes the app fast
def prepare_data():
    train = pd.read_csv('Titanic_train.csv')
    
    # Simple cleaning (adjust these if your notebook used different features)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    
    # Filling missing values
    train['Age'] = train['Age'].fillna(train['Age'].median())
    train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
    
    X = train[features]
    y = train['Survived']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model, features

model, features = prepare_data()

# --- 3. UI FOR TESTING ---
st.header("Test with your Data")
uploaded_file = st.file_uploader("Upload your testing CSV file", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    
    # Apply the same cleaning to the test data
    test_proc = test_data.copy()
    test_proc['Age'] = test_proc['Age'].fillna(28) # Default age
    test_proc['Sex'] = test_proc['Sex'].map({'male': 0, 'female': 1})
    
    # Predict
    predictions = model.predict(test_proc[features])
    test_data['Survived_Prediction'] = predictions
    
    st.subheader("Results")
    st.write(test_data[['Name', 'Survived_Prediction']])
    
    # Download Button
    csv = test_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "results.csv", "text/csv")

else:
    st.info("Please upload a CSV file to see predictions.")