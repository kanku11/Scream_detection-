import numpy as np
import librosa
from keras.models import load_model

# Paths to model and classes
MODEL_PATH = "scream_model.h5"       # Make sure this file exists in your project directory
CLASSES_PATH = "classes.npy"         # This should contain the list of class names

# Load the model
model = load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load class labels (ensure allow_pickle=True)
classes = np.load(CLASSES_PATH, allow_pickle=True)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def predict_scream(audio_path):
    features = extract_features(audio_path)
    features = features.reshape(1, -1)  # reshape for model input
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]
    return predicted_label
