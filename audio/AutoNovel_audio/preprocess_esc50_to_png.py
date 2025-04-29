import os
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

ESC50_ROOT = "/mainfs/scratch/dk2g21/AutoNovel/datasets/ESC-50"
OUTPUT_DIR = "/mainfs/scratch/dk2g21/AutoNovel/datasets/ESC-50-spectrograms/clean"
SAMPLE_RATE = 22050
N_MELS = 128

def save_mel_png(wav_path, label, out_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        resampler = T.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    mel_spec = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS
    )(waveform)

    mel_spec = mel_spec.squeeze().log2().clamp(min=-10, max=10)

    os.makedirs(out_path, exist_ok=True)
    file_id = os.path.basename(wav_path).replace(".wav", ".png")
    full_path = os.path.join(out_path, file_id)

    plt.imsave(full_path, mel_spec.numpy(), cmap="magma")

def main():
    meta_path = os.path.join(ESC50_ROOT, "meta/esc50.csv")
    audio_dir = os.path.join(ESC50_ROOT, "audio")
    df = pd.read_csv(meta_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fold = f"fold{row['fold']}"
        label = row["category"]
        filename = row["filename"]
        wav_path = os.path.join(audio_dir, filename)
        out_path = os.path.join(OUTPUT_DIR, label)
        save_mel_png(wav_path, label, out_path)

if __name__ == "__main__":
    main()
