import random
import cv2
import numpy as np

from .config import DegradationConfig
from .transforms import (
    apply_blur,
    apply_noise,
    apply_jpeg_compression,
    apply_fading,
    apply_scratches,
    apply_missing_regions,
    apply_resolution_degradation
)

def degrade_image(img, config: DegradationConfig, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    damaged = img.copy()

    # The order of degradations matters somewhat. We will apply them in this order.
    # Resolution -> Scratches -> Missing -> Fading -> Blur -> Noise -> JPEG
    
    res_cfg = config.get('resolution')
    if res_cfg.get('enabled') and random.random() < res_cfg.get('probability'):
        sev = random.uniform(*res_cfg.get('severity', [0.2, 1.0]))
        damaged = apply_resolution_degradation(damaged, sev)

    scratch_cfg = config.get('scratches')
    if scratch_cfg.get('enabled') and random.random() < scratch_cfg.get('probability'):
        sev = random.uniform(*scratch_cfg.get('severity', [0.2, 1.0]))
        damaged = apply_scratches(damaged, sev)

    missing_cfg = config.get('missing_regions')
    if missing_cfg.get('enabled') and random.random() < missing_cfg.get('probability'):
        sev = random.uniform(*missing_cfg.get('severity', [0.2, 1.0]))
        damaged = apply_missing_regions(damaged, sev)

    fading_cfg = config.get('fading')
    if fading_cfg.get('enabled') and random.random() < fading_cfg.get('probability'):
        sev = random.uniform(*fading_cfg.get('severity', [0.2, 1.0]))
        damaged = apply_fading(damaged, sev)

    blur_cfg = config.get('blur')
    if blur_cfg.get('enabled') and random.random() < blur_cfg.get('probability'):
        sev = random.uniform(*blur_cfg.get('severity', [0.2, 1.0]))
        damaged = apply_blur(damaged, sev)

    noise_cfg = config.get('noise')
    if noise_cfg.get('enabled') and random.random() < noise_cfg.get('probability'):
        sev = random.uniform(*noise_cfg.get('severity', [0.2, 1.0]))
        damaged = apply_noise(damaged, sev)

    jpeg_cfg = config.get('jpeg')
    if jpeg_cfg.get('enabled') and random.random() < jpeg_cfg.get('probability'):
        sev = random.uniform(*jpeg_cfg.get('severity', [0.2, 1.0]))
        damaged = apply_jpeg_compression(damaged, sev)

    return damaged
