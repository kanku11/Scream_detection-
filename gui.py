import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load your trained model
model = load_model('scream_model.h5')

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_scream(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0][0]

def browse_file():
    global selected_file
    selected_file = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )
    if selected_file:
        label_file.config(text=selected_file)
        label_result.config(text="Processing...", fg="blue")
        root.update_idletasks()  # refresh UI
        try:
            prob = predict_scream(selected_file)
            if prob > 0.5:
                result_text = f"Scream Detected! (Confidence: {prob:.2f})"
                label_result.config(text=result_text, fg="red")
            else:
                result_text = f"No Scream Detected. (Confidence: {prob:.2f})"
                label_result.config(text=result_text, fg="green")
            btn_play.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{str(e)}")
            label_result.config(text="Prediction failed.", fg="red")
            btn_play.config(state=tk.DISABLED)

def play_audio():
    try:
        pygame.mixer.music.load(selected_file)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Error", f"Audio playback failed:\n{str(e)}")

# Main window setup
root = tk.Tk()
root.title("Human Scream Detection")
root.geometry("550x300")
root.configure(bg="#1e1e2f")

# Fonts and colors
font_title = ("Helvetica", 18, "bold")
font_button = ("Helvetica", 12)
font_label = ("Helvetica", 10)
color_bg = "#1e1e2f"
color_fg = "#e0e0e0"
color_button = "#4a90e2"
color_button_hover = "#357ABD"

# Title label
label_title = tk.Label(root, text="Human Scream Detection", font=font_title, bg=color_bg, fg=color_fg)
label_title.pack(pady=(20,10))

# Frame for file selection and buttons
frame_controls = tk.Frame(root, bg=color_bg)
frame_controls.pack(pady=10)

btn_browse = tk.Button(frame_controls, text="Select Audio File", font=font_button, bg=color_button, fg="white",
                       activebackground=color_button_hover, activeforeground="white", command=browse_file, width=18)
btn_browse.grid(row=0, column=0, padx=10)

btn_play = tk.Button(frame_controls, text="Play Audio", font=font_button, bg=color_button, fg="white",
                     activebackground=color_button_hover, activeforeground="white", command=play_audio, width=18, state=tk.DISABLED)
btn_play.grid(row=0, column=1, padx=10)

# Label to show selected file path
label_file = tk.Label(root, text="No file selected", font=font_label, bg=color_bg, fg=color_fg, wraplength=500, justify="center")
label_file.pack(pady=(15, 5))

# Label to show prediction result
label_result = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg=color_bg, fg="yellow")
label_result.pack(pady=10)

# Run the app
root.mainloop()
