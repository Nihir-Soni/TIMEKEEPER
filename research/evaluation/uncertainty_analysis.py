import os
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from research.uncertainty import UncertaintyInferencer
from research.calibration import UncertaintyCalibrator

def extract_regional_means(map_array, patch_size=64):
    """
    Splits the map_array into patch_size x patch_size regions
    and computes the mean for each region.
    """
    h, w = map_array.shape
    means = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = map_array[y:y+patch_size, x:x+patch_size]
            means.append(np.mean(patch))
    return np.array(means)

def run_analysis(data_dir, checkpoint_path, calibrator_path, out_dir):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    inferencer = UncertaintyInferencer(checkpoint_path)
    calibrator = UncertaintyCalibrator.load(calibrator_path)
    
    samples = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    
    all_pixel_preds = []
    all_pixel_actuals = []
    all_region_preds = []
    all_region_actuals = []
    
    print(f"Analyzing {len(samples)} samples...")
    for sample_dir in tqdm(samples):
        d_path = sample_dir / "damaged.png"
        r_path = sample_dir / "restored.png"
        e_path = sample_dir / "actual_error.npy"
        
        if not (d_path.exists() and r_path.exists()):
            continue
            
        damaged_bgr = cv2.imread(str(d_path))
        restored_bgr = cv2.imread(str(r_path))
        
        # 1. Infer raw uncertainty
        raw_unc = inferencer.infer(damaged_bgr, restored_bgr)
        np.save(sample_dir / "raw_uncertainty.npy", raw_unc)
        
        # 2. Calibration
        calib_unc = calibrator.calibrate(raw_unc)
        
        # 3. Confidence Mapping
        # Mapping: high uncertainty (high predicted error) -> low confidence
        # Maximum possible error in uint8 space is 255
        confidence = np.clip(1.0 - (calib_unc / 255.0), 0.0, 1.0)
        
        # Save Calibrated Confidence visual
        # Confidence: 1.0 (high) -> 0.0 (low)
        # Convert to 0-255 for colormap mapping
        conf_vis = (confidence * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(sample_dir / "calibrated_confidence.png"), heatmap)
        
        # 4. Regional Evaluation (if GT actual error exists)
        if e_path.exists():
            actual_err = np.load(str(e_path))
            
            if calib_unc.shape != actual_err.shape:
                calib_unc = cv2.resize(calib_unc, (actual_err.shape[1], actual_err.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Pixel-wise subsampling for global correlation (avoid memory explosion)
            mask = np.random.rand(*calib_unc.shape) < 0.05
            all_pixel_preds.append(calib_unc[mask])
            all_pixel_actuals.append(actual_err[mask])
            
            # Regional means
            patch_size = 64
            reg_preds = extract_regional_means(calib_unc, patch_size=patch_size)
            reg_actuals = extract_regional_means(actual_err, patch_size=patch_size)
            
            all_region_preds.append(reg_preds)
            all_region_actuals.append(reg_actuals)
            
    # Calculate Correlations
    results = {}
    if len(all_pixel_preds) > 0:
        pixel_preds_flat = np.concatenate(all_pixel_preds)
        pixel_actuals_flat = np.concatenate(all_pixel_actuals)
        
        reg_preds_flat = np.concatenate(all_region_preds)
        reg_actuals_flat = np.concatenate(all_region_actuals)
        
        p_pearson, _ = pearsonr(pixel_preds_flat, pixel_actuals_flat)
        p_spearman, _ = spearmanr(pixel_preds_flat, pixel_actuals_flat)
        
        r_pearson, _ = pearsonr(reg_preds_flat, reg_actuals_flat)
        r_spearman, _ = spearmanr(reg_preds_flat, reg_actuals_flat)
        
        results = {
            "pixel_pearson": float(p_pearson),
            "pixel_spearman": float(p_spearman),
            "regional_pearson": float(r_pearson),
            "regional_spearman": float(r_spearman),
            "num_regions_analyzed": len(reg_preds_flat)
        }
        
        # Generate Reliability / Calibration curves (Plots)
        plt.figure(figsize=(8, 6))
        plt.scatter(reg_preds_flat, reg_actuals_flat, alpha=0.3, s=10)
        plt.plot([0, 255], [0, 255], 'r--', label='Ideal Calibration')
        plt.xlabel('Mean Calibrated Predicted Uncertainty (Error)')
        plt.ylabel('Mean Actual Restoration Error')
        plt.title('Regional Reliability Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / "regional_reliability_scatter.png")
        plt.close()
        
    with open(out_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Analysis complete. Results saved to {out_dir}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to research_restored_output")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_uncertainty_model.pth")
    parser.add_argument("--calibrator", type=str, required=True, help="Path to calibrator.pkl")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dir for analysis")
    
    args = parser.parse_args()
    os.chdir(Path(__file__).parent.parent.parent)
    run_analysis(args.data_dir, args.checkpoint, args.calibrator, args.out_dir)
