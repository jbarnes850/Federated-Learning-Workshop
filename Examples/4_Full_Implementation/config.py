"""
Network Configuration Module
==========================

This module defines the configuration settings for the food security network,
including node setup, data handling, model deployment, and security parameters.

Progress Tracking:
----------------
- Basic Configuration ✓
- Security Settings ✓
- Network Roles ✓
- Federation Setup ✓

Validation Steps:
---------------
1. Verify environment variables
2. Check security settings
3. Validate network roles
4. Test federation config

Usage:
-----
Load network configuration:
    config = NetworkConfig.load_config()
    print(f"Network Name: {config.NETWORK_NAME}")
"""

import os
import logging
from cryptography.fernet import Fernet
import aivm_client as aic
from aivm_client.models import get_supported_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkConfig:
    """Configuration class for food security network setup."""
    
    def __init__(self):
        """Initialize network configuration with validation."""
        self.progress = {
            "env_setup": False,
            "security_config": False,
            "roles_defined": False,
            "federation_ready": False
        }
        
        try:
            # Network identification
            self.NETWORK_NAME = "FoodSecurityNetwork"
            self.NODE_ID = os.getenv('NODE_ID', 'default_node')
            
            # Nillion Network Configuration
            self.NILLION_NETWORK_URL = os.getenv('NILLION_NETWORK_URL', 'http://localhost:8080')
            self.NILLION_NODE_KEY = os.getenv('NILLION_NODE_KEY')
            self.NILLION_NETWORK_KEY = os.getenv('NILLION_NETWORK_KEY')
            
            # Data security
            self.DATA_ENCRYPTION_KEY = Fernet.generate_key()
            self.DATA_STORAGE_PATH = "data_storage"
            
            # AIVM configuration
            self.AIVM_API_KEY = os.getenv('AIVM_API_KEY')
            if not self.AIVM_API_KEY:
                raise ValueError("AIVM_API_KEY not set")
            self.AIVM_ENDPOINT = os.getenv(
                'AIVM_ENDPOINT',
                'https://aivm.example.com/api'
            )
            
            # Model configuration
            self.MODEL_NAME = "FoodSecurityBERT"
            self.MODEL_VERSION = "v1.0"
            self.LOCAL_MODEL_PATH = os.path.join(
                self.DATA_STORAGE_PATH,
                f"{self.MODEL_NAME}_{self.NODE_ID}.pth"
            )
            
            # Network roles and permissions
            self.ROLES = {
                'data_provider': ['read', 'encrypt'],
                'model_deployer': ['read', 'write'],
                'aggregator': ['read']
            }
            
            # Security settings
            self.COMMUNICATION_PROTOCOL = "HTTPS"
            self.SSL_CERTIFICATE = os.path.join(
                self.DATA_STORAGE_PATH,
                "ssl_certificate.crt"
            )
            
            # Federation configuration
            self.FEDERATION_ENABLED = True
            self.FEDERATION_STRATEGY = 'average'
            
            self._validate_configuration()
            
        except Exception as e:
            logger.error(f"Configuration initialization failed: {e}")
            raise
    
    def _validate_configuration(self):
        """Validate all configuration settings."""
        try:
            # Validate environment setup
            assert self.NODE_ID, "Node ID not set"
            assert self.AIVM_API_KEY, "AIVM API key not set"
            self.progress["env_setup"] = True
            
            # Validate security configuration
            assert self.DATA_ENCRYPTION_KEY, "Encryption key not generated"
            assert self.COMMUNICATION_PROTOCOL == "HTTPS", "Insecure protocol"
            self.progress["security_config"] = True
            
            # Validate roles
            assert all(role in self.ROLES for role in [
                'data_provider', 'model_deployer', 'aggregator'
            ]), "Missing required roles"
            self.progress["roles_defined"] = True
            
            # Validate federation setup
            assert isinstance(self.FEDERATION_ENABLED, bool)
            assert self.FEDERATION_STRATEGY in ['average', 'weighted']
            self.progress["federation_ready"] = True
            
            logger.info("✓ Configuration validated successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    @staticmethod
    def load_config():
        """Load and validate configuration settings."""
        return NetworkConfig()
    
    def get_progress(self):
        """Get current progress status of configuration setup."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }

if __name__ == "__main__":
    try:
        config = NetworkConfig.load_config()
        logger.info("\nConfiguration Status:")
        for step, status in config.get_progress().items():
            logger.info(f"{step}: {status}")
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")

