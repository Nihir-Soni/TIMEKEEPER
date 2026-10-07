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
from .degrade import degrade_image
