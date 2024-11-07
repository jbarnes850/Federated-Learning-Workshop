"""
Food Security Network Implementation
==================================

This module implements a secure and privacy-preserving network for food security analysis
using Nillion AIVM. It handles encrypted data sharing and federated predictions across
multiple nodes while maintaining data privacy.

Prerequisites:
------------
- AIVM devnet must be running (see README.md)
- Environment setup completed
- Model deployed

Progress Tracking:
----------------
- Network Initialization ✓
- Secure Prediction ✓
- Data Sharing ✓
- Privacy Verification ✓

Usage:
-----
1. Start AIVM devnet in a separate terminal:
   aivm-devnet

2. Initialize and use the network:
   network = FoodSecurityNetwork()
   prediction = await network.predict_demand(local_data)
"""

import aivm_client as aic
import logging
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from config import NetworkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodSecurityNetwork:
    """Implementation of a secure food security analysis network."""
    
    def __init__(self):
        """Initialize network components with validation."""
        self.progress = {
            "network_init": False,
            "client_setup": False,
            "encryption_ready": False,
            "prediction_ready": False,
            "privacy_verified": False
        }
        
        try:
            # Load configuration
            self.config = NetworkConfig.load_config()
            
            # Initialize AIVM client
            self.client = aic.Client()
            self.progress["client_setup"] = True
            logger.info("✓ Connected to AIVM devnet")
            
            # Initialize encryption
            self.cipher = Fernet(self.config.DATA_ENCRYPTION_KEY)
            self.progress["encryption_ready"] = True
            
            # Verify network setup
            self._verify_network_setup()
            self.progress["network_init"] = True
            logger.info("✓ Network initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Network initialization failed: {e}")
            raise

    async def predict_demand(self, local_data: str) -> Optional[Dict[str, Any]]:
        """
        Make privacy-preserved demand predictions.
        
        Args:
            local_data: Raw local data for prediction
            
        Returns:
            dict: Prediction results
        """
        try:
            # Tokenize and encrypt data
            tokenized_data = aic.tokenize(local_data)
            encrypted_data = aic.BertTinyCryptensor(*tokenized_data)
            
            # Get prediction
            prediction = aic.get_prediction(
                encrypted_data,
                self.config.MODEL_NAME
            )
            
            self.progress["prediction_ready"] = True
            logger.info("✓ Secure prediction completed")
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

    def share_insights(self, insights_data: str) -> Optional[bytes]:
        """
        Share insights securely across the network.
        
        Args:
            insights_data: Data to be shared
            
        Returns:
            bytes: Encrypted shared data
        """
        try:
            # Encrypt insights
            encrypted_insights = self.cipher.encrypt(
                insights_data.encode()
            )
            
            # Share through AIVM
            shared_data = self.client.share_encrypted_data(
                encrypted_insights
            )
            
            logger.info("✓ Insights shared securely")
            return shared_data
            
        except Exception as e:
            logger.error(f"❌ Sharing failed: {e}")
            raise

    def _verify_network_setup(self) -> None:
        """Verify network configuration and privacy."""
        try:
            # Verify model availability
            models = aic.get_supported_models()
            if self.config.MODEL_NAME not in models:
                raise ValueError(f"Model {self.config.MODEL_NAME} not found")
            
            # Verify privacy preservation
            privacy_checks = [
                hasattr(self, 'cipher'),
                self.config.SSL_ENABLED or self.config.NODE_HOST == 'localhost',
                bool(self.config.DATA_ENCRYPTION_KEY)
            ]
            
            if all(privacy_checks):
                self.progress["privacy_verified"] = True
                logger.info("✓ Privacy verification complete")
            else:
                raise ValueError("Privacy requirements not met")
                
        except Exception as e:
            logger.error(f"❌ Network verification failed: {e}")
            raise

    def get_progress(self) -> Dict[str, str]:
        """Get current progress status."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        try:
            # Initialize network
            network = FoodSecurityNetwork()
            
            # Test prediction
            test_data = "Need food assistance for family of 4"
            prediction = await network.predict_demand(test_data)
            
            # Test sharing
            insights = "Monthly demand increased by 25%"
            shared = network.share_insights(insights)
            
            # Show progress
            logger.info("\nNetwork Status:")
            for step, status in network.get_progress().items():
                logger.info(f"{step}: {status}")
                
            logger.info("✓ Network test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Network test failed: {e}")
            exit(1)

    asyncio.run(main())
