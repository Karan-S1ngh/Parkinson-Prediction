import streamlit as st
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import joblib
import os
import tempfile

MODEL_DIR = 'model_assets'
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Use st.cache_resource to load these only once and speed up the app
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        imputer = joblib.load(os.path.join(MODEL_DIR, 'imputer.joblib'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
        return model, scaler, imputer, label_encoder, feature_names
    except FileNotFoundError:
        st.error(f"Error: Model assets not found in '{MODEL_DIR}'. Please run the training script first.")
        return None, None, None, None, None

# Load all assets
model, scaler, imputer, label_encoder, feature_names = load_assets()

def extract_all_features(file_path):
    """
    Extracts a comprehensive set of voice features from an audio file.
    This function mirrors the feature extraction process used for training.
    """
    # Initialize all features to NaN
    features = {f'mfcc_{i+1}': np.nan for i in range(40)}
    extra_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                      'rms_energy', 'mean_pitch', 'jitter_local']
    for f in extra_features:
        features[f] = np.nan

    # Librosa Features
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        for i in range(40):
            features[f'mfcc_{i+1}'] = mfccs[i]

        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
    except Exception as e:
        print(f"Could not process Librosa features for {os.path.basename(file_path)}: {e}")

    # Parselmouth Features
    try:
        sound = parselmouth.Sound(file_path)
        pitch = sound.to_pitch()
        features['mean_pitch'] = call(pitch, "Get mean", 0, 0, "Hertz")
        
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        n_pulses = call(point_process, "Get number of points")
        if n_pulses >= 2:
            features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception as e:
        # This will result in NaN for parselmouth features if something goes wrong
        pass

    return features

# STREAMLIT APP INTERFACE
st.set_page_config(page_title="Parkinson's Voice Analysis", layout="wide")
st.title("🔬 Parkinson's Disease Prediction via Voice Analysis")
st.write("Upload a short voice recording (e.g., sustaining the 'aah' sound) to analyze for potential signs of Parkinson's Disease.")

# Sidebar for file upload
st.sidebar.header("Upload Audio File")
uploaded_file = st.sidebar.file_uploader("Choose a WAV or MP3 file", type=['wav', 'mp3'])

if model is None:
    st.warning("Application is not ready. Please ensure model assets are available.")
else:
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Analyze button
        if st.button("Analyze Voice Sample"):
            with st.spinner("Processing audio and running analysis..."):
                try:
                    # Save uploaded file to a temporary location to get a file path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_filepath = tmp.name

                    # 1. Extract features from the uploaded audio file
                    extracted_features = extract_all_features(temp_filepath)
                    
                    # 2. Prepare features for the model (must match training steps)
                    # Convert to DataFrame in the correct order
                    features_df = pd.DataFrame([extracted_features], columns=feature_names)
                    
                    # Impute missing values using the trained imputer
                    features_imputed = imputer.transform(features_df)
                    
                    # Scale features using the trained scaler
                    features_scaled = scaler.transform(features_imputed)

                    # 3. Make Prediction
                    prediction_proba = model.predict_proba(features_scaled)
                    prediction = model.predict(features_scaled)
                    
                    # 4. Decode and Display Results
                    predicted_label = label_encoder.inverse_transform(prediction)[0]
                    confidence_score = prediction_proba[0][prediction[0]] * 100

                    st.subheader("Analysis Complete")
                    col1, col2 = st.columns(2)

                    with col1:
                        if predicted_label.lower() == 'parkinson':
                            st.error(f"**Prediction:** {predicted_label}")
                        else:
                            st.success(f"**Prediction:** {predicted_label}")

                    with col2:
                        st.info(f"**Confidence:** {confidence_score:.2f}%")
                    
                    st.write("---")
                    
                    # Explainability: Show feature importance or values
                    with st.expander("View Extracted Voice Features"):
                        st.dataframe(features_df)
                        
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                finally:
                    # Clean up the temporary file
                    if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                        os.remove(temp_filepath)