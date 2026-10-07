import os
import torch
import cv2
import numpy as np
from pathlib import Path

from .model import UncertaintyUNet
from .config import UncertaintyConfig

class UncertaintyInferencer:
    def __init__(self, checkpoint_path, config_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = UncertaintyConfig(config_path)
        
        self.model = UncertaintyUNet(
            in_channels=self.config.get('model', 'in_channels'),
            out_channels=self.config.get('model', 'out_channels'),
            base_filters=self.config.get('model', 'base_filters')
        ).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"[UncertaintyInferencer] Warning loading checkpoint {checkpoint_path}: {e}")
        self.model.eval()
        self.image_size = self.config.get('dataset', 'image_size')

    def infer(self, damaged_bgr, restored_bgr):
        orig_h, orig_w = damaged_bgr.shape[:2]
        
        # Preprocess for neural net
        d_resized = cv2.resize(damaged_bgr, (self.image_size, self.image_size))
        r_resized = cv2.resize(restored_bgr, (self.image_size, self.image_size))
        
        d_rgb = cv2.cvtColor(d_resized, cv2.COLOR_BGR2RGB)
        r_rgb = cv2.cvtColor(r_resized, cv2.COLOR_BGR2RGB)
        
        d_tensor = torch.from_numpy(d_rgb).float().permute(2, 0, 1) / 255.0
        r_tensor = torch.from_numpy(r_rgb).float().permute(2, 0, 1) / 255.0
        
        input_tensor = torch.cat([d_tensor, r_tensor], dim=0).unsqueeze(0).to(self.device)
        
        # Infer from neural model
        with torch.no_grad():
            pred = self.model(input_tensor)
            
        pred_np = pred.squeeze().cpu().numpy()
        pred_err = pred_np * 255.0
        raw_uncertainty = cv2.resize(pred_err, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        raw_uncertainty = np.clip(raw_uncertainty, 0, None)

        # ── Calculate Multi-scale Structural & Feature Difference ─────────────
        r_bgr_resized = cv2.resize(restored_bgr, (orig_w, orig_h))
        
        # 1. Full BGR Color Channel L1 difference
        diff_color = np.mean(np.abs(damaged_bgr.astype(np.float32) - r_bgr_resized.astype(np.float32)), axis=2)
        
        # 2. Gradient / Edge structure difference
        d_gray = cv2.cvtColor(damaged_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r_gray = cv2.cvtColor(r_bgr_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        sobel_d = cv2.Sobel(d_gray, cv2.CV_32F, 1, 1)
        sobel_r = cv2.Sobel(r_gray, cv2.CV_32F, 1, 1)
        diff_grad = np.abs(sobel_d - sobel_r)
        
        # 3. High-frequency detail density (Laplacian for complex textures)
        lap_r = np.abs(cv2.Laplacian(r_gray, cv2.CV_32F))
        
        struct_unc = diff_color + 0.5 * diff_grad + 0.1 * lap_r

        # Blend neural model output with structural metric
        if raw_uncertainty.std() > 0.5:
            final_uncertainty = raw_uncertainty + struct_unc
        else:
            final_uncertainty = struct_unc
            
        return final_uncertainty

def save_raw_uncertainty(raw_uncertainty, out_dir):
    out_dir = Path(out_dir)
    np.save(out_dir / "raw_uncertainty.npy", raw_uncertainty)
    
    vis_max = raw_uncertainty.max()
    if vis_max > 0:
        vis = (raw_uncertainty / vis_max) * 255.0
    else:
        vis = raw_uncertainty
        
    cv2.imwrite(str(out_dir / "raw_uncertainty.png"), vis.astype(np.uint8))

