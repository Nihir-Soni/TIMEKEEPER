import os
import sys
import cv2
import json
import csv
import shutil
import argparse
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

from research.evaluation import calculate_error_map, save_error_map, calculate_psnr, calculate_ssim

def run_restoration_pipeline(input_dir, output_dir):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples = sorted([p for p in in_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    print(f"Found {len(samples)} samples to process.")
    
    temp_input = out_dir / "temp_input"
    temp_output = out_dir / "temp_output"
    
    if temp_input.exists(): shutil.rmtree(temp_input)
    if temp_output.exists(): shutil.rmtree(temp_output)
    
    temp_input.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy all damaged images to temp folder with unique names
    for sample_dir in samples:
        damaged_path = sample_dir / "damaged.png"
        if damaged_path.exists():
            shutil.copy(damaged_path, temp_input / f"{sample_dir.name}.png")
            
    # 2. Run existing Global restoration model
    # It requires python executable to be the same one running this script
    python_exe = sys.executable
    cmd = [
        python_exe, "Global/test.py",
        "--test_mode", "Full",
        "--Quality_restore",
        "--test_input", str(temp_input),
        "--outputs_dir", str(temp_output),
        "--gpu_ids", "-1" # Default to CPU to ensure it runs anywhere, users can change to 0 if needed
    ]
    
    # Check if GPU is available (optional, but let's stick to -1 to be safe unless specified, or we check torch)
    import torch
    if torch.cuda.is_available():
        cmd[-1] = "0"
        
    print(f"Running existing restoration pipeline with command:\n{' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Restoration pipeline failed with error:\n{result.stderr}")
        
    # 3. Process outputs, calculate metrics and error maps
    metrics_list = []
    success_count = 0
    failed_count = 0
    
    for sample_dir in tqdm(samples, desc="Evaluating Results"):
        sample_name = sample_dir.name
        sample_out_dir = out_dir / sample_name
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        
        gt_path = sample_dir / "ground_truth.png"
        damaged_path = sample_dir / "damaged.png"
        restored_path = temp_output / "restored_image" / f"{sample_name}.png"
        
        # Copy damaged and gt to the output dir for completeness
        if gt_path.exists(): shutil.copy(gt_path, sample_out_dir / "ground_truth.png")
        if damaged_path.exists(): shutil.copy(damaged_path, sample_out_dir / "damaged.png")
        
        if not restored_path.exists():
            print(f"Failed to find restored image for {sample_name}")
            failed_count += 1
            continue
            
        # Move restored
        final_restored = sample_out_dir / "restored.png"
        shutil.copy(restored_path, final_restored)
        
        # Evaluate
        gt_img = cv2.imread(str(gt_path))
        restored_img = cv2.imread(str(final_restored))
        
        if gt_img is None or restored_img is None:
            failed_count += 1
            continue
            
        # Actual error map
        err_map = calculate_error_map(restored_img, gt_img)
        
        # Sanity check
        if np.isnan(err_map).any() or np.isinf(err_map).any() or err_map.min() < 0:
            print(f"Numerical error in {sample_name} error map!")
            
        save_error_map(err_map, str(sample_out_dir / "actual_error.npy"), str(sample_out_dir / "actual_error.png"))
        
        psnr_val = calculate_psnr(restored_img, gt_img)
        ssim_val = calculate_ssim(restored_img, gt_img)
        
        metrics_list.append({
            "sample": sample_name,
            "psnr": psnr_val,
            "ssim": ssim_val
        })
        
        # Comparison image
        damaged_img = cv2.imread(str(damaged_path))
        err_vis = cv2.imread(str(sample_out_dir / "actual_error.png"))
        
        # Resize all to GT shape just for visualization
        h, w = gt_img.shape[:2]
        r_img = cv2.resize(restored_img, (w, h))
        d_img = cv2.resize(damaged_img, (w, h))
        e_img = cv2.resize(err_vis, (w, h))
        
        comp = np.hstack([gt_img, d_img, r_img, e_img])
        cv2.imwrite(str(sample_out_dir / "comparison.png"), comp)
        
        success_count += 1
        
    # Generate final reports
    if len(metrics_list) > 0:
        mean_psnr = np.mean([m['psnr'] for m in metrics_list])
        median_psnr = np.median([m['psnr'] for m in metrics_list])
        mean_ssim = np.mean([m['ssim'] for m in metrics_list])
        median_ssim = np.median([m['ssim'] for m in metrics_list])
        
        summary = {
            "samples": len(metrics_list),
            "mean_psnr": float(mean_psnr),
            "median_psnr": float(median_psnr),
            "mean_ssim": float(mean_ssim),
            "median_ssim": float(median_ssim),
            "lpips": "not enabled"
        }
        
        with open(out_dir / "metrics.json", 'w') as f:
            json.dump(summary, f, indent=4)
            
        with open(out_dir / "metrics.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["sample", "psnr", "ssim"])
            writer.writeheader()
            for row in metrics_list:
                writer.writerow(row)
                
        print(f"Successfully processed {success_count} samples. Failed: {failed_count}")
        print(f"Mean PSNR: {mean_psnr:.2f} dB, Mean SSIM: {mean_ssim:.4f}")
        
    # Cleanup temp
    if temp_input.exists(): shutil.rmtree(temp_input)
    if temp_output.exists(): shutil.rmtree(temp_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run restoration and calculate evaluation metrics.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with generated synthetic pairs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restoration results and metrics")
    
    args = parser.parse_args()
    
    # Need to run from project root because of Global/test.py imports
    os.chdir(Path(__file__).parent.parent.parent)
    run_restoration_pipeline(args.input_dir, args.output_dir)
