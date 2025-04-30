#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration handler for FLPSF Editor
"""

import os
import logging
import yaml
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger('flpsf_editor.config')

class Config:
    """
    Configuration handler that loads and provides access to application settings
    """
    
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        self.config_path = config_path
        self.config_data = {}
        self.load()
        
    def load(self) -> bool:
        """
        Load configuration from YAML file
        
        Returns:
            bool: True if configuration loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.config_path):
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
                
            with open(self.config_path, 'r') as file:
                self.config_data = yaml.safe_load(file)
                
            logger.info(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
            
    def save(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to YAML file
        
        Args:
            config_path (str, optional): Path to save configuration file. 
                                        If None, uses original path.
                                        
        Returns:
            bool: True if configuration saved successfully, False otherwise
        """
        save_path = config_path or self.config_path
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as file:
                yaml.dump(self.config_data, file, default_flow_style=False)
                
            logger.info(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path
        
        Args:
            path (str): Path to configuration value using dot notation (e.g., "ui.theme")
            default (Any): Default value to return if path not found
            
        Returns:
            Any: Configuration value or default if not found
        """
        parts = path.split('.')
        value = self.config_data
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, path: str, value: Any) -> bool:
        """
        Set configuration value by path
        
        Args:
            path (str): Path to configuration value using dot notation (e.g., "ui.theme")
            value (Any): Value to set
            
        Returns:
            bool: True if value set successfully, False otherwise
        """
        parts = path.split('.')
        config = self.config_data
        
        try:
            # Navigate to the containing dictionary
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
                
            # Set the value
            config[parts[-1]] = value
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration at {path}: {str(e)}")
            return False
            
    def get_all(self) -> Dict:
        """
        Get all configuration data
        
        Returns:
            Dict: Complete configuration data
        """
        return self.config_data.copy()