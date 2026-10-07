import yaml

class UncertaintyConfig:
    def __init__(self, config_path=None, config_dict=None):
        self.config = {
            'model': {
                'in_channels': 6, # damaged (3) + restored (3)
                'out_channels': 1,
                'base_filters': 32
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 1e-4,
                'epochs': 100,
                'save_dir': 'research_checkpoints',
                'patience': 10 # For early stopping
            },
            'dataset': {
                'image_size': 256,
                'train_split': 0.8
            }
        }
        
        if config_dict:
            self.config.update(config_dict)
        elif config_path:
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self.config.update(loaded)

    def get(self, section, key=None):
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
