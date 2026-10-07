import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

from .config import DegradationConfig
from .degrade import degrade_image

def generate_dataset(input_dir, output_dir, config_path=None, seed=42):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = DegradationConfig(config_path=config_path)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = [p for p in in_dir.rglob('*') if p.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Generating Dataset")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
            
        sample_seed = seed + idx if seed is not None else None
        damaged = degrade_image(img, config, seed=sample_seed)
        
        # Save paired structure
        sample_name = f"sample_{idx:04d}"
        sample_dir = out_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(sample_dir / "ground_truth.png"), img)
        cv2.imwrite(str(sample_dir / "damaged.png"), damaged)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetically degraded dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with clean images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save paired dataset")
    parser.add_argument("--config", type=str, default=None, help="YAML config file for degradations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    generate_dataset(args.input_dir, args.output_dir, args.config, args.seed)
