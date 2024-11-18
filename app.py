import streamlit as st
import torch
import torchaudio
from io import BytesIO
import numpy as np
import pickle

# Load the pre-trained model from the uploaded file
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error("Model could not be loaded. Please check the model file.")
else:
    st.success("Model loaded successfully.")

# Function to preprocess the audio file for the model
def preprocess_audio(uploaded_file):
    try:
        # Read the uploaded file using BytesIO
        file_bytes = BytesIO(uploaded_file.read())
        
        # Load the audio using torchaudio
        waveform, sample_rate = torchaudio.load(file_bytes)
        st.write(f"Original Sample Rate: {sample_rate}, Waveform Shape: {waveform.shape}")
        
        waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono and add batch dimension
        if sample_rate != 32000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
            waveform = resampler(waveform)
        st.write(f"Resampled Waveform Shape: {waveform.shape}")
        return waveform
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Function to make predictions using the loaded model
def predict_real_or_fake(waveform):
    try:
        model.eval()
        with torch.no_grad():
            output = model(waveform)
            st.write(f"Model Output: {output}")
            
            # If the model output is not a single value, adjust accordingly
            if isinstance(output, torch.Tensor):
                prediction = torch.sigmoid(output).item()
            else:
                prediction = output
            
        return "Fake" if prediction > 0.5 else "Real"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit App
st.title("Audio Deepfake Detection App")
st.write("Upload an audio file (.wav) to check if it is real or fake.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Preprocess and predict
    waveform = preprocess_audio(uploaded_file)
    
    if waveform is not None:
        prediction = predict_real_or_fake(waveform)
        if prediction is not None:
            # Display the result
            st.write(f"Prediction: The audio is **{prediction}**.")
        else:
            st.error("Prediction could not be made.")
