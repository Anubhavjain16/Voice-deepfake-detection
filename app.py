import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import pickle

# Load the pre-trained model
model_path = 'best_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to extract audio features
def extract_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        st.error(f"Error in processing audio: {e}")
        return None

# Streamlit UI
st.title("Fake or Real Audio Detector")
st.write("Upload an audio file to check if it's real or fake.")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features from the uploaded file
    features = extract_features("uploaded_audio.wav")
    
    if features is not None:
        # Reshape features for prediction
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        result = "Real" if prediction[0] == 1 else "Fake"
        
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Failed to extract features from the audio file.")
else:
    st.info("Please upload a WAV file to proceed.")
