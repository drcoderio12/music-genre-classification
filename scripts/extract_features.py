import os
import numpy as np
import pandas as pd
import librosa

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)

        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)

        # Combine features into a single array
        return np.hstack([mfccs_mean, chroma_mean, contrast_mean])
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None  # Return None if there was an error

# Main function to extract features from all audio files in a directory
def extract_all_features(directory):
    # Create the features directory if it does not exist
    features_dir = 'C:/MIni Projects/DS/MGC Project/features'
    os.makedirs(features_dir, exist_ok=True)

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    features = []
    labels = []

    # Iterate over genre directories
    for genre_dir in os.listdir(directory):
        genre_path = os.path.join(directory, genre_dir)

        if not os.path.isdir(genre_path):  # Ensure it's a directory
            continue
        
        # Iterate over audio files in the genre directory
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):  # Process only WAV files
                file_path = os.path.join(genre_path, file)
                print(f"Processing file: {file_path}")  # Debugging line

                # Extract features
                data = extract_features(file_path)
                if data is not None:  # Check if features were extracted successfully
                    features.append(data)
                    labels.append(genre_dir)

    # Create a DataFrame to hold features and labels
    features_df = pd.DataFrame(features)
    features_df['label'] = labels
    
    # Save to the created directory
    features_df.to_csv(os.path.join(features_dir, 'extracted_features.csv'), index=False)  
    print(f"Features extracted and saved to {os.path.join(features_dir, 'extracted_features.csv')}")

if __name__ == "__main__":
    extract_all_features("C:/MIni Projects/DS/MGC Project/data/genre")  # Update path to the correct genres directory
