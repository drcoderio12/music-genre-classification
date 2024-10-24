from flask import Flask, request, render_template
import joblib
import numpy as np
import librosa
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

# Function to extract features from the audio file
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
    contrast_mean = np.mean(contrast.T, axis=0)
    
    return np.hstack([mfccs_mean, chroma_mean, contrast_mean])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        # Save the file to a temporary location
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)

        # Extract features and predict the genre
        features = extract_features(file_path).reshape(1, -1)
        predicted_genre = model.predict(features)

        # Clean up the temporary file
        os.remove(file_path)

        return render_template('index.html', genre=predicted_genre[0])
    
    return render_template('index.html', genre=None)

if __name__ == '__main__':
    app.run(debug=True)
