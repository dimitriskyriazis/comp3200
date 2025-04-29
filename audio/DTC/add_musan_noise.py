import os
import random
import librosa
import numpy as np
import soundfile as sf
from glob import glob

def add_noise_to_audio(clean, noise, snr_db):
    clean_len = len(clean)
    noise = np.tile(noise, int(np.ceil(clean_len / len(noise))))[:clean_len]
    noise = noise / np.std(noise)
    clean = clean / np.std(clean)
    snr_linear = 10 ** (snr_db / 10)
    noise_scaling = np.sqrt(np.sum(clean ** 2) / (snr_linear * np.sum(noise ** 2)))
    noisy = clean + noise_scaling * noise
    return noisy / np.max(np.abs(noisy))

def mix_esc50_with_musan(clean_dir, musan_noise_dir, output_root, snrs=[15,10,5]):
    clean_files = glob(os.path.join(clean_dir, "*.wav"))
    musan_files = glob(os.path.join(musan_noise_dir, "**", "*.wav"), recursive=True)

    for snr in snrs:
        print(f"Generating noisy data at {snr}dB SNR...")
        out_dir = os.path.join(output_root, f"snr{snr}")
        os.makedirs(out_dir, exist_ok=True)

        for file in clean_files:
            try:
                clean_audio, sr = librosa.load(file, sr=None)
                noise_audio, _ = librosa.load(random.choice(musan_files), sr=sr)
                noisy_audio = add_noise_to_audio(clean_audio, noise_audio, snr)

                out_path = os.path.join(out_dir, os.path.basename(file))
                sf.write(out_path, noisy_audio, sr)
            except Exception as e:
                print(f"Failed to process {file}: {e}")

if __name__ == "__main__":
    mix_esc50_with_musan(
        clean_dir="/mainfs/scratch/dk2g21/DTC/datasets/ESC-50/audio/",
        musan_noise_dir="/mainfs/scratch/dk2g21/DTC/datasets/musan/musan/noise/",
        output_root="/mainfs/scratch/dk2g21/DTC/datasets/ESC50-noisy-wavs/",
        snrs=[15, 10, 5]
    )
