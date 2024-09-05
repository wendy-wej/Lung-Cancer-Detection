import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from io import BytesIO

# Function to load the model from Hugging Face
def load_symptom_model():
    # Replace with your actual Hugging Face model URL
    symptom_model_url = "https://huggingface.co/Chinwendu/lung_cancer_image_model/resolve/main/lung_cancer_symptom_model.pkl"
        
    # Send a request to download the model file
    response = requests.get(symptom_model_url)
    if response.status_code == 200:
        # Write the content to a file
        with open('lung_cancer_symptom_model.pkl', 'wb') as f:
            f.write(response.content)
        
        # Load the model using pickle
        with open('lung_cancer_symptom_model.pkl', 'rb') as f:
            symptom_model = pickle.load(f)
        
        return symptom_model
    else:
        raise Exception(f"Failed to download the model, status code: {response.status_code}")

def load_ct_model():
    ct_model_url = 'https://huggingface.co/Chinwendu/lung_ct_detection_model/resolve/main/lung_ct_model.keras'
            

    response = requests.get(ct_model_url)
    if response.status_code == 200:
        with open('lung_ct_model.keras', 'wb') as f:
            f.write(response.content)
        ct_model = tf.keras.models.load_model('lung_ct_model.keras')
        return ct_model
    else:
        raise Exception(f"Failed to download the model, status code: {response.status_code}")

# Load the model
symptom_model = load_symptom_model()
ct_model = load_ct_model()

# Title and description
st.title("Lung Cancer Prediction App")

# Encoding dictionary for user inputs
yes_no_encoding = {"No": 0, "Yes": 1}
with st.expander("Lung Cancer Symptom Prediction", expanded=True):
    st.write("This app predicts the likelihood of lung cancer based on user inputs.")
    # User inputs with Yes/No options
    yellow_fingers = st.selectbox('Yellow Fingers', ['No', 'Yes'], help='Select "Yes" if you have yellow fingers, otherwise select "No"')
    anxiety = st.selectbox('Anxiety', ['No', 'Yes'], help='Select "Yes" if you have anxiety, otherwise select "No"')
    peer_pressure = st.selectbox('Peer Pressure', ['No', 'Yes'], help='Select "Yes" if you experience peer pressure, otherwise select "No"')
    chronic_disease = st.selectbox('Chronic Disease', ['No', 'Yes'], help='Select "Yes" if you have a chronic disease, otherwise select "No"')
    fatigue = st.selectbox('Fatigue', ['No', 'Yes'], help='Select "Yes" if you experience fatigue, otherwise select "No"')
    allergy = st.selectbox('Allergy', ['No', 'Yes'], help='Select "Yes" if you have allergies, otherwise select "No"')
    wheezing = st.selectbox('Wheezing', ['No', 'Yes'], help='Select "Yes" if you experience wheezing, otherwise select "No"')
    alcohol_consuming = st.selectbox('Alcohol Consuming', ['No', 'Yes'], help='Select "Yes" if you consume alcohol, otherwise select "No"')
    coughing = st.selectbox('Coughing', ['No', 'Yes'], help='Select "Yes" if you have a cough, otherwise select "No"')
    swallowing_difficulty = st.selectbox('Swallowing Difficulty', ['No', 'Yes'], help='Select "Yes" if you have difficulty swallowing, otherwise select "No"')
    chest_pain = st.selectbox('Chest Pain', ['No', 'Yes'], help='Select "Yes" if you experience chest pain, otherwise select "No"')

# Encode inputs for the model
    input_features = [
        yes_no_encoding[yellow_fingers],
        yes_no_encoding[anxiety],
        yes_no_encoding[peer_pressure],
        yes_no_encoding[chronic_disease],
        yes_no_encoding[fatigue],
        yes_no_encoding[allergy],
        yes_no_encoding[wheezing],
        yes_no_encoding[alcohol_consuming],
        yes_no_encoding[coughing],
        yes_no_encoding[swallowing_difficulty],
        yes_no_encoding[chest_pain]
    ]
    
    # Collect inputs into a DataFrame
    input_data = pd.DataFrame([input_features], columns=[
        'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
        'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
        'COUGHING', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'
    ])
    
    # Prediction
    if st.button("Predict", key="symptom_model"):
        try:
            symptom_prediction = symptom_model.predict(input_data)
            if symptom_prediction[0] == 1:
                st.error("The model predicts a high likelihood of lung cancer.")
            else:
                st.success("The model predicts a low likelihood of lung cancer.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        
st.header("CT Scan-based Lung Cancer Prediction")  
with st.expander("CT Scan-based Lung Cancer Prediction", expanded=True):
    uploaded_file = st.file_uploader("Upload a lung CT scan image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded CT Scan', use_column_width=True)
    
        # Preprocess the image
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (256, 256))
        img_scaled = img_resized / 255.0
        img_reshaped = np.expand_dims(img_scaled, axis=0)
    
        # Predict using the CT scan model
        if st.button("Predict", key="ct_model"):
            try:
                ct_prediction = ct_model.predict(img_reshaped)
                class_names = ['Benign', 'Malignant', 'Normal']
                ct_predicted_class = class_names[np.argmax(ct_prediction)]
                st.write(f"Prediction: {ct_predicted_class}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")