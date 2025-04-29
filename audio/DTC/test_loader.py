# test_loader.py
from data.imagenetloader import ImageNetLoader30

def main():
    # Change the path below to your ESC-50 spectrogram folder (for example, snr5)
    data_dir = "datasets/ESC50-noisy-spectrograms/snr5"
    
    # Using num_workers=0 for safety on HPC
    loader = ImageNetLoader30(batch_size=16, path=data_dir, aug=None, shuffle=False, num_workers=0)
    
    print("[TEST] Dataset size:", len(loader.dataset))
    
    # Try iterating over a few batches
    for i, (x, y, idx) in enumerate(loader):
        print(f"[TEST] Batch {i}: x shape = {x.shape}, y = {y}")
        if i >= 1:
            break

if __name__ == "__main__":
    main()
