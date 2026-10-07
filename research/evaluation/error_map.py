import cv2
import numpy as np

def calculate_error_map(restored, ground_truth):
    """
    Calculate the actual restoration error map as E(x,y) = |I_R(x,y) - I_GT(x,y)|
    Reduces to a single magnitude channel by averaging across RGB.
    """
    if restored.shape != ground_truth.shape:
        # Resize restored to ground_truth shape if necessary (log this in caller)
        restored = cv2.resize(restored, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    res_f = restored.astype(np.float32)
    gt_f = ground_truth.astype(np.float32)
    
    # E(x,y) = |I_R(x,y) - I_GT(x,y)|
    abs_err = np.abs(res_f - gt_f)
    
    # Mean across channels for a spatial error map
    if len(abs_err.shape) == 3:
        spatial_err = np.mean(abs_err, axis=2)
    else:
        spatial_err = abs_err
        
    return spatial_err

def save_error_map(spatial_err, npy_path, png_path):
    """
    Save the raw numerical array and a normalized visual PNG representation.
    """
    # Save numerical
    np.save(npy_path, spatial_err)
    
    # Save visual (normalized to 0-255 for visibility)
    # Be careful: if max is 0, we don't divide by 0
    err_max = spatial_err.max()
    if err_max > 0:
        vis_err = (spatial_err / err_max) * 255.0
    else:
        vis_err = spatial_err
        
    cv2.imwrite(png_path, vis_err.astype(np.uint8))
