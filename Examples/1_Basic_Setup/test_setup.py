"""
Basic Setup Validation Script
============================

This script provides comprehensive testing for the AIVM setup and validates the environment 
configuration before proceeding with the workshop exercises.

Progress Tracking:
----------------
- Environment Setup ✓
- AIVM Client Configuration ✓
- Model Support Verification ✓
- Basic Connectivity Test ✓

Validation Steps:
---------------
1. Verify environment variables
2. Test AIVM client initialization
3. Check supported models
4. Validate model upload capability
5. Test basic prediction functionality

Usage:
-----
Run this script after completing the basic installation script in install.sh:
    python test_setup.py
"""

import unittest
import logging
from nillion_aivm import Client as aic
from nillion_aivm.models import get_supported_models
import os
import sys
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestAIVM(unittest.TestCase):
    def setUp(self):
        """Initialize AIVM client and validate environment."""
        logger.info("Starting AIVM setup validation...")
        
        # Track setup progress
        self.progress = {
            "env_setup": False,
            "client_init": False,
            "model_support": False,
            "upload_capability": False
        }
        
        # Initialize client (no API key needed for devnet)
        try:
            self.client = aic()
            # No need to configure API key for devnet
            logger.info("✓ AIVM client initialized successfully")
            self.progress["client_init"] = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize AIVM client: {e}")
            raise

        # Add environment validation
        self._validate_python_version()
        self._validate_dependencies()

        # Verify supported models
        models = get_supported_models()
        if "LeNet5MNIST" not in models and "BertTiny" not in models:
            raise ValueError("Required models not supported")

    def _validate_python_version(self):
        """Verify Python version compatibility."""
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        logger.info("✓ Python version validated")

    def _validate_dependencies(self):
        """Verify all required packages are installed."""
        required_packages = [
            'nillion-aivm',
            'torch',
            'transformers',
            'pandas',
            'cryptography'
        ]
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✓ {package} installed")
            except ImportError:
                raise ImportError(f"Required package {package} not installed")

    def test_get_supported_models(self):
        """Verify that required models are supported."""
        try:
            models = self.client.get_supported_models()
            self.assertIn("LeNet5MNIST", models)
            logger.info("✓ Model support verification passed")
            self.progress["model_support"] = True
        except Exception as e:
            logger.error(f"❌ Model support verification failed: {e}")
            raise

    def test_upload_lenet5_model(self):
        """Test model upload and prediction capabilities."""
        try:
            model_path = "path/to/your/model"
            self.client.upload_lenet5_model(model_path, "MyCustomLeNet5")
            input_data = "path/to/your/input"
            prediction = self.client.get_prediction(input_data, "MyCustomLeNet5")
            self.assertIsNotNone(prediction)
            logger.info("✓ Model upload and prediction test passed")
            self.progress["upload_capability"] = True
        except Exception as e:
            logger.error(f"❌ Model upload test failed: {e}")
            raise

    def tearDown(self):
        """Report final setup status."""
        logger.info("\nSetup Validation Summary:")
        for step, status in self.progress.items():
            logger.info(f"{step}: {'✓' if status else '❌'}")

def test_aivm_connection():
    # List available models
    models = aic.get_supported_models()
    print(f"Available models: {models}")
    return "LeNet5MNIST" in models or "BertTiny" in models

def test_network_config():
    """Verify network configuration."""
    try:
        network_key = os.getenv('NILLION_NETWORK_KEY')
        node_key = os.getenv('NILLION_NODE_KEY')
        network_url = os.getenv('NILLION_NETWORK_URL', 'http://localhost:8080')
        
        assert network_key, "Network key not set"
        assert node_key, "Node key not set"
        assert network_url, "Network URL not set"
        
        logger.info("✓ Network configuration verified")
        return True
    except Exception as e:
        logger.error(f"❌ Network configuration failed: {e}")
        raise

if __name__ == "__main__":
    unittest.main()

