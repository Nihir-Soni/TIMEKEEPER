import time
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

from GUI import modify
from research.uncertainty import UncertaintyInferencer
from research.calibration import UncertaintyCalibrator

def run_ablation(test_image_path):
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        return
        
    print(f"Running Ablation Study on {test_image_path}...")
    
    # Pre-load models for uncertainty to be fair
    inferencer = UncertaintyInferencer("research_checkpoints/best_uncertainty_model.pth")
    calibrator = UncertaintyCalibrator.load("research_checkpoints/calibrator.pkl")
    orig_img = cv2.imread(test_image_path)
    
    # We will use the GUI modify function to run the heavy pipelines,
    # and time them exactly as the GUI would.
    
    results = []
    
    # Setup temp dir
    temp_dir = project_root / "ablation_temp_input"
    temp_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy(test_image_path, temp_dir / "test.png")
    
    def dummy_status(msg): pass
    
    # 1. Restoration only
    start_time = time.time()
    modify(
        image_filename=str(temp_dir),
        with_scratch=False,
        with_colorize=False,
        with_hr=False,
        status_callback=dummy_status,
        saturation=1.0
    )
    t1 = time.time() - start_time
    results.append({"config": "Restoration only", "time": t1})
    
    # 2. Restoration + Colorization
    start_time = time.time()
    modify(
        image_filename=str(temp_dir),
        with_scratch=False,
        with_colorize=True,
        with_hr=False,
        status_callback=dummy_status,
        saturation=1.0
    )
    t2 = time.time() - start_time
    results.append({"config": "Restoration + Colorization", "time": t2})
    
    # Load the output image to pass to uncertainty models
    restored_path = project_root / "output" / "final_output" / "test.png"
    if not restored_path.exists():
        print("Failed to find restored output!")
        return
    restored_img = cv2.imread(str(restored_path))
    
    # 3. Restoration + Colorization + Raw Uncertainty
    start_time = time.time()
    # The first part is roughly t2
    raw_unc = inferencer.infer(orig_img, restored_img)
    t3_unc_overhead = time.time() - start_time
    results.append({"config": "Restoration + Colorization + Raw Uncertainty", "time": t2 + t3_unc_overhead})
    
    # 4. Restoration + Colorization + Calibrated Uncertainty
    start_time = time.time()
    calib_unc = calibrator.calibrate(raw_unc)
    confidence = np.clip(1.0 - (calib_unc / 255.0), 0.0, 1.0)
    conf_vis = (confidence * 255).astype(np.uint8)
    conf_heatmap = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
    t4_calib_overhead = time.time() - start_time
    results.append({"config": "Restoration + Colorization + Calibrated Uncertainty", "time": t2 + t3_unc_overhead + t4_calib_overhead})
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n--- Ablation Results ---")
    print(f"{'Configuration':<60} | {'Time (s)':<10}")
    print("-" * 75)
    for res in results:
        print(f"{res['config']:<60} | {res['time']:.2f}")
    print("-" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    args = parser.parse_args()
    run_ablation(args.image)
