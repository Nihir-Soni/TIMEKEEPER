import os
import pickle
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:
    IsotonicRegression = None

class UncertaintyCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        if self.method == 'isotonic':
            if IsotonicRegression is not None:
                self.model = IsotonicRegression(out_of_bounds='clip')
            else:
                self.model = None
        else:
            raise ValueError(f"Calibration method {method} not supported yet.")
            
    def fit(self, raw_uncertainty_1d, actual_error_1d):
        """
        Fit the calibrator. 
        Input arrays should be flattened 1D numpy arrays of all pixels in the validation set.
        """
        if self.model is not None:
            self.model.fit(raw_uncertainty_1d, actual_error_1d)
        
    def calibrate(self, raw_uncertainty_map):
        """
        Transform a raw uncertainty map (2D or 3D) into a calibrated map.
        """
        shape = raw_uncertainty_map.shape
        flat = raw_uncertainty_map.flatten()
        if self.model is not None and hasattr(self.model, 'f_'):
            try:
                calibrated_flat = self.model.predict(flat)
                return calibrated_flat.reshape(shape)
            except Exception:
                pass
        # Fallback: percentile dynamic normalization
        p98 = np.percentile(raw_uncertainty_map, 98)
        p02 = np.percentile(raw_uncertainty_map, 2)
        if p98 > p02:
            return np.clip((raw_uncertainty_map - p02) / (p98 - p02) * 255.0, 0, 255)
        max_v = raw_uncertainty_map.max()
        if max_v > 0:
            return (raw_uncertainty_map / max_v) * 255.0
        return raw_uncertainty_map
        
    def save(self, filepath):
        if joblib is not None:
            joblib.dump(self, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        
    @classmethod
    def load(cls, filepath):
        if os.path.exists(filepath):
            try:
                if joblib is not None:
                    return joblib.load(filepath)
                else:
                    with open(filepath, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                print(f"[UncertaintyCalibrator] Warning loading {filepath}: {e}")
        return cls()


