import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_img = image.copy()
    h, w = noisy_img.shape
    num_salt = int(salt_prob * h * w)
    num_pepper = int(pepper_prob * h * w)

    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_img.shape]
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_img.shape]

    noisy_img[salt_coords[0], salt_coords[1]] = 255
    noisy_img[pepper_coords[0], pepper_coords[1]] = 0
    return noisy_img

def process_images(input_root, output_root, noise_type, std, sp_ratio):
    for root, _, files in os.walk(input_root):
        for fname in tqdm(files):
            if not fname.endswith(".png"):
                continue
            class_dir = os.path.relpath(root, input_root)
            in_path = os.path.join(root, fname)
            out_dir = os.path.join(output_root, class_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)

            img = Image.open(in_path).convert("L")
            img_np = np.array(img)

            if noise_type == "gaussian":    
                noisy = add_gaussian_noise(img_np, std=std)
            elif noise_type == "saltpepper":
                noisy = add_salt_pepper_noise(img_np, salt_prob=sp_ratio, pepper_prob=sp_ratio)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            cv2.imwrite(out_path, noisy)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--noise_type", type=str, choices=["gaussian", "saltpepper"], required=True)
    parser.add_argument("--std", type=float, default=25)
    parser.add_argument("--sp_ratio", type=float, default=0.01)

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.noise_type, args.std, args.sp_ratio)

if __name__ == "__main__":
    main()
