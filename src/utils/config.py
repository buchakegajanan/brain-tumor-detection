"""
Configuration Management Module
Loads and manages project configuration from YAML file
"""

import yaml
import os
from pathlib import Path

class Config:
    """Configuration class to load and access project settings"""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize configuration
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, *keys, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            *keys: Nested keys to access configuration
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get('data', 'batch_size')  # Returns 32
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def create_directories(self):
        """Create necessary project directories if they don't exist"""
        directories = [
            self.get('data', 'raw_data_path'),
            self.get('data', 'processed_data_path'),
            self.get('paths', 'models'),
            self.get('paths', 'results'),
            self.get('paths', 'plots'),
            self.get('paths', 'logs'),
            self.get('paths', 'uploads'),
            self.get('gradcam', 'output_path')
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __repr__(self):
        """String representation"""
        return f"Config(config_path='{self.config_path}')"

# Global configuration instance
config = Config()
