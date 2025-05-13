🎙️ Voice Deepfake Detection System

🧠 Overview

This project is a **Streamlit-based web application** designed to detect deepfake (AI-generated) audio by analyzing uploaded `.wav` files. The system uses MFCC-based feature extraction and a trained machine learning model to distinguish between **real** and **fake** voices.

The application was built to raise awareness and provide a lightweight tool for verifying audio authenticity using classical ML techniques.

---

🚀 Features

- Upload and analyze `.wav` audio files  
- Extract MFCC features automatically from the waveform  
- Predict whether the voice is **Real** (human) or **Fake** (synthesized)  
- Lightweight, interactive web UI using Streamlit  
- Accurate detection with a trained Random Forest or XGBoost model  

---

🔧 Technologies Used

- **Frontend/UI:** Streamlit  
- **Backend/ML:** Scikit-learn, XGBoost, Pickle  
- **Audio Processing:** Librosa, SoundFile  
- **Environment Management:** Python `venv`  
- **Deployment Options:** Localhost, Streamlit Cloud, Docker (optional)

---

 🛠️ Installation

 1. Clone the repository
```bash
git clone https://github.com/your-username/voice-deepfake-detector.git
cd voice-deepfake-detector
````

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Navigate to the app in your browser at:
📍 `http://localhost:8501`

---

## 🎯 How It Works

1. The user uploads a `.wav` audio file.
2. The application extracts **MFCC** features using `librosa`.
3. These features are passed to a **trained ML model** (`best_model.pkl`).
4. The system outputs whether the voice is likely **Real** or **Fake**.

---

## 🎯 API Logic (Internal - via app.py)

* **Upload File**
  Saves the uploaded `.wav` audio file locally as `uploaded_audio.wav`.

* **Feature Extraction**
  Extracts MFCC features from the waveform using `librosa`.

* **Model Prediction**
  Loads a pre-trained model and classifies the voice.

* **UI Display**
  Presents the result as `Real` or `Fake` in the browser.

---

## 🧪 Model Details

* **Input Features:** MFCC mean vectors from `.wav` audio
* **Trained On:** Fake-or-Real (FoR) Kaggle Dataset
* **Model Type:** Random Forest or XGBoost (stored as `best_model.pkl`)
* **Accuracy:** >92% on test data

---

## 📁 File Structure

```
voice-deepfake-detector/
├── app.py                # Streamlit app
├── best_model.pkl        # Trained ML model
├── requirements.txt      # Python dependencies
├── uploaded_audio.wav    # Saved user audio (temporary)
├── README.md             # Project documentation
```

---

## 🧠 Dataset

* **Source:** [Fake-or-Real Voice Dataset (Kaggle)](https://www.kaggle.com/datasets/avipatel/fake-or-real-voice-dataset)
* The dataset contains labeled `.wav` files for **real** and **fake** samples used during model training.

---

## 🌐 Deployment Options

* **Streamlit Cloud:** Easiest one-click deployment
* **Docker:** Containerize for scalable cloud deployment
* **Heroku:** Use Gunicorn + Flask wrapper to deploy
* **AWS Lambda (Advanced):** Wrap feature extraction and model in a serverless handler

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit your changes
4. Push the branch and open a Pull Request

---

## 📫 Contact

**Author:** Anubhav Jain
📧 [anubhavjain.1116@gmail.com](mailto:anubhavjain.1116@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/anubhav-jain1/)
💻 [GitHub](https://github.com/Anubhavjain16)

---

## 📄 License

This project is intended for academic and research use only.
For commercial or enterprise use, please contact the author.
