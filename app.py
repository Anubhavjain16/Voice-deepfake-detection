import streamlit as st
import torch
import torchaudio
import numpy as np
import pickle

# Load the pre-trained model from the uploaded file
@st.cache_resource
def load_model():
    with open('/mnt/data/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Function to preprocess the audio file for the model
def preprocess_audio(file):
    waveform, sample_rate = torchaudio.load(file)
    waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono and add batch dimension
    if sample_rate != 32000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
        waveform = resampler(waveform)
    return waveform

# Function to make predictions using the loaded model
def predict_real_or_fake(waveform):
    model.eval()
    with torch.no_grad():
        output = model.predict(waveform)
        prediction = torch.sigmoid(output).item()
    return "Fake" if prediction > 0.5 else "Real"

# Streamlit App
st.title("Audio Deepfake Detection App")
st.write("Upload an audio file (.wav) to check if it is real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Preprocess and predict
    waveform = preprocess_audio(uploaded_file)
    prediction = predict_real_or_fake(waveform)
    
    # Display the result
    st.write(f"Prediction: The audio is **{prediction}**.")
