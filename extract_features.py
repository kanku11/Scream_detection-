import librosa
import os
import numpy as np
import pandas as pd

# Get current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Updated path handling
DATA_PATH = os.path.join(BASE_DIR, "audio_data")
FEATURES_PATH = os.path.join(BASE_DIR, "features.csv")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_dataset():
    features = []
    for label in ['scream', 'non_scream']:
        dir_path = os.path.join(DATA_PATH, label)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}. Skipping...")
            continue

        for file in os.listdir(dir_path):
            if file.endswith(".wav"):
                file_path = os.path.join(dir_path, file)
                data = extract_features(file_path)
                if data is not None:
                    features.append([*data, label])

    df = pd.DataFrame(features)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Feature extraction completed. Data saved to {FEATURES_PATH}")

if __name__ == "__main__":
    process_dataset()
