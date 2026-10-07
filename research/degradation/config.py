import yaml

class DegradationConfig:
    def __init__(self, config_path=None, config_dict=None):
        self.config = {
            'blur': {'enabled': True, 'probability': 0.5, 'severity': [0.2, 1.0]},
            'noise': {'enabled': True, 'probability': 0.5, 'severity': [0.2, 1.0]},
            'jpeg': {'enabled': True, 'probability': 0.5, 'severity': [0.2, 1.0]},
            'fading': {'enabled': True, 'probability': 0.3, 'severity': [0.2, 1.0]},
            'scratches': {'enabled': True, 'probability': 0.4, 'severity': [0.2, 1.0]},
            'missing_regions': {'enabled': True, 'probability': 0.2, 'severity': [0.2, 1.0]},
            'resolution': {'enabled': True, 'probability': 0.4, 'severity': [0.2, 1.0]}
        }
        
        if config_dict:
            self.config.update(config_dict)
        elif config_path:
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self.config.update(loaded)

    def get(self, key):
        return self.config.get(key, {'enabled': False})
