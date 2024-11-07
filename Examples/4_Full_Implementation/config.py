"""
Network Configuration Module
==========================

This module defines configuration settings for the food security network,
including node setup, data handling, model deployment, and security parameters.

Prerequisites:
------------
- AIVM devnet must be running
- Environment variables set
- Dependencies installed

Progress Tracking:
----------------
- Environment Setup ✓
- Security Configuration ✓
- Network Configuration ✓
- Model Configuration ✓

Usage:
-----
Load network configuration:
    config = NetworkConfig.load_config()
"""

import os
import logging
import aivm_client as aic
from cryptography.fernet import Fernet
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkConfig:
    """Configuration class for food security network setup."""
    
    def __init__(self):
        """Initialize network configuration with validation."""
        self.progress = {
            "env_setup": False,
            "security_config": False,
            "network_config": False,
            "model_config": False
        }
        
        try:
            # Verify AIVM connection
            self.client = aic.Client()
            logger.info("✓ Connected to AIVM devnet")
            
            # Network identification
            self.NETWORK_NAME = "FoodSecurityNetwork"
            self.NODE_ID = os.getenv('NODE_ID', 'default_node')
            
            # Network settings
            self.NODE_HOST = os.getenv('NODE_HOST', 'localhost')
            self.NODE_PORT = int(os.getenv('NODE_PORT', '50050'))
            
            # Security settings
            self.DATA_ENCRYPTION_KEY = Fernet.generate_key()
            self.DATA_STORAGE_PATH = "data_storage"
            self.SSL_ENABLED = os.getenv('SSL_ENABLED', 'false').lower() == 'true'
            
            # Model configuration
            self.MODEL_NAME = "FoodSecurityBERT"
            self.MODEL_VERSION = "v1.0"
            self.MODEL_PATH = os.path.join(
                self.DATA_STORAGE_PATH,
                f"{self.MODEL_NAME}_{self.NODE_ID}.pth"
            )
            
            self._validate_configuration()
            
        except Exception as e:
            logger.error(f"❌ Configuration initialization failed: {e}")
            raise
    
    def _validate_configuration(self) -> None:
        """Validate all configuration settings."""
        try:
            # Validate environment
            self._validate_environment()
            self.progress["env_setup"] = True
            
            # Validate security
            self._validate_security()
            self.progress["security_config"] = True
            
            # Validate network
            self._validate_network()
            self.progress["network_config"] = True
            
            # Validate model
            self._validate_model()
            self.progress["model_config"] = True
            
            logger.info("✓ Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _validate_environment(self) -> None:
        """Validate environment setup."""
        required_vars = ['NODE_ID', 'NODE_HOST', 'NODE_PORT']
        for var in required_vars:
            if not os.getenv(var):
                logger.warning(f"⚠️ {var} not set, using default")
    
    def _validate_security(self) -> None:
        """Validate security configuration."""
        if not self.DATA_ENCRYPTION_KEY:
            raise ValueError("Encryption key not generated")
        os.makedirs(self.DATA_STORAGE_PATH, exist_ok=True)
    
    def _validate_network(self) -> None:
        """Validate network configuration."""
        if not isinstance(self.NODE_PORT, int):
            raise ValueError("Invalid port number")
    
    def _validate_model(self) -> None:
        """Validate model configuration."""
        models = aic.get_supported_models()
        if "BertTiny" not in models:
            raise ValueError("Required model not supported")
    
    def get_progress(self) -> Dict[str, str]:
        """Get current progress status."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }

    @staticmethod
    def load_config():
        """Load and validate configuration settings."""
        return NetworkConfig()

if __name__ == "__main__":
    try:
        config = NetworkConfig.load_config()
        logger.info("\nConfiguration Status:")
        for step, status in config.get_progress().items():
            logger.info(f"{step}: {status}")
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        exit(1)

