import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def wav_to_png(input_dir, output_dir, sr=22050, dpi=100):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = glob(os.path.join(input_dir, "*.wav"))

    for wav_path in wav_files:
        y, sr = librosa.load(wav_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        label = wav_path.split("-")[-1].split(".")[0]  # e.g., 0 from "1-100032-A-0.wav"
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        png_path = os.path.join(label_dir, os.path.basename(wav_path).replace(".wav", ".png"))
        plt.figure(figsize=(2, 2))
        librosa.display.specshow(S_dB, sr=sr, fmax=8000)
        plt.axis('off')
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()

if __name__ == "__main__":
    for snr in [15, 10, 5]:
        wav_path = f"datasets/ESC50-noisy-wavs/snr{snr}"
        png_path = f"datasets/ESC50-noisy-spectrograms/snr{snr}"
        wav_to_png(wav_path, png_path)
