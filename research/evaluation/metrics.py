import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

def calculate_psnr(restored, ground_truth):
    if restored.shape != ground_truth.shape:
        restored = cv2.resize(restored, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    mse = np.mean((restored.astype(np.float32) - ground_truth.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return float(psnr)

def calculate_ssim(restored, ground_truth):
    if restored.shape != ground_truth.shape:
        restored = cv2.resize(restored, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    # skimage ssim expects channels_last for color images
    win_size = min(7, restored.shape[0], restored.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
        
    if win_size < 3:
        return 0.0 # Image too small
        
    score = ssim_metric(ground_truth, restored, win_size=win_size, channel_axis=2 if len(restored.shape)==3 else None)
    return float(score)

def calculate_lpips(restored, ground_truth):
    # Optional LPIPS. Return None if not enabled/installed.
    return None
