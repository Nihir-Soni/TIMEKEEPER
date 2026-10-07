import time
import numpy as np
import cv2
import torch
import os
from pathlib import Path

from research.uncertainty import UncertaintyInferencer
from research.calibration import UncertaintyCalibrator

def run_audit_ablation():
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    inferencer = UncertaintyInferencer("research_checkpoints/best_uncertainty_model.pth")
    calibrator = UncertaintyCalibrator.load("research_checkpoints/calibrator.pkl")
    
    # 512x512 image represents a standard resolution patch
    img1 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Warmup
    raw = inferencer.infer(img1, img2)
    _ = calibrator.calibrate(raw)
    
    unc_times = []
    calib_times = []
    
    for _ in range(10):
        t0 = time.perf_counter()
        raw = inferencer.infer(img1, img2)
        t1 = time.perf_counter()
        cal = calibrator.calibrate(raw)
        t2 = time.perf_counter()
        
        unc_times.append(t1 - t0)
        calib_times.append(t2 - t1)
        
    print(f"Uncertainty Inference Time (512x512): {np.mean(unc_times)*1000:.2f} ms +/- {np.std(unc_times)*1000:.2f} ms")
    print(f"Calibration Time (512x512): {np.mean(calib_times)*1000:.2f} ms +/- {np.std(calib_times)*1000:.2f} ms")

if __name__ == "__main__":
    run_audit_ablation()
