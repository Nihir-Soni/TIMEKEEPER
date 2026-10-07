import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from research.uncertainty import UncertaintyInferencer, save_raw_uncertainty
from research.calibration.calibrator import UncertaintyCalibrator

def fit_calibration(data_dir, checkpoint_path, out_dir):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Instantiate inferencer
    inferencer = UncertaintyInferencer(checkpoint_path)
    
    # We must use validation data.
    # To mimic what dataset.py does for validation split:
    all_samples = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    np.random.seed(42)
    
    # In python 3, random.shuffle is inplace. dataset.py used np.random.shuffle.
    # For exactly replicating dataset.py splitting:
    samples_array = np.array(all_samples)
    np.random.shuffle(samples_array)
    
    split_idx = int(len(samples_array) * 0.8)
    val_samples = samples_array[split_idx:]
    
    if len(val_samples) == 0 and len(samples_array) > 0:
        val_samples = samples_array # Fallback if too few samples
        
    print(f"Fitting calibrator using {len(val_samples)} validation samples...")
    
    raw_unc_list = []
    actual_err_list = []
    
    for sample_dir in tqdm(val_samples, desc="Processing val samples"):
        d_path = sample_dir / "damaged.png"
        r_path = sample_dir / "restored.png"
        e_path = sample_dir / "actual_error.npy"
        
        if not (d_path.exists() and r_path.exists() and e_path.exists()):
            continue
            
        damaged_bgr = cv2.imread(str(d_path))
        restored_bgr = cv2.imread(str(r_path))
        actual_err = np.load(str(e_path))
        
        # 1. Infer raw uncertainty
        raw_unc = inferencer.infer(damaged_bgr, restored_bgr)
        
        # 2. Save raw uncertainty for phase 7 requirements
        save_raw_uncertainty(raw_unc, sample_dir)
        
        # We need raw_unc and actual_err to match shape exactly
        if raw_unc.shape != actual_err.shape:
            raw_unc = cv2.resize(raw_unc, (actual_err.shape[1], actual_err.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        # Optional: Subsample pixels if there are too many (e.g., millions of pixels per image)
        # Using 10% of pixels to fit the isotonic regression to save memory/time
        mask = np.random.rand(*raw_unc.shape) < 0.1
        
        raw_unc_list.append(raw_unc[mask])
        actual_err_list.append(actual_err[mask])
        
    if len(raw_unc_list) == 0:
        print("No valid data found to fit calibration!")
        return
        
    # Concatenate all
    X = np.concatenate(raw_unc_list)
    y = np.concatenate(actual_err_list)
    
    print(f"Fitting Isotonic Regression on {len(X)} pixels...")
    calibrator = UncertaintyCalibrator(method='isotonic')
    calibrator.fit(X, y)
    
    calib_path = out_dir / "calibrator.pkl"
    calibrator.save(calib_path)
    print(f"Calibrator saved to {calib_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit uncertainty calibration on validation set.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to research_restored_output")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_uncertainty_model.pth")
    parser.add_argument("--out_dir", type=str, default="research_checkpoints", help="Output dir for calibrator.pkl")
    
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent.parent.parent)
    fit_calibration(args.data_dir, args.checkpoint, args.out_dir)
