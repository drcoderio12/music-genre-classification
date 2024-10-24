import os
import numpy as np
import pandas as pd
import joblib
import librosa

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        print(f"Loading audio file from: {file_path}")  # Debugging line
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        return np.hstack([mfccs_mean, chroma_mean, contrast_mean])
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Function to predict genre
def predict_genre(file_path, model):
    features = extract_features(file_path)
    if features is not None:
        features = features.reshape(1, -1)  # Reshape for model prediction
        predicted_genre = model.predict(features)
        return predicted_genre[0]
    else:
        print("No features extracted. Cannot predict genre.")
        return None

# Load the model
model_path = 'C:/MIni Projects/DS/MGC Project/models/model.pkl'
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

if model is not None:
    # Specify the directory to scan for audio files
    directory = 'C:/MIni Projects/DS/MGC Project/data/genre/blues'
    
    # Checking the files in the directory
    try:
        files = os.listdir(directory)
        print(f"Files in {directory}:")
        
        for f in files:
            file_path = os.path.join(directory, f)  # Create the full file path
            predicted_genre = predict_genre(file_path, model)
            
            if predicted_genre is not None:
                print(f'Predicted genre for {f}: {predicted_genre}')
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
else:
    print("Model could not be loaded. Exiting the script.")
