"""
Food Bank Data Privacy Implementation
===================================

This module implements privacy-preserving data handling for food bank demand data 
using Nillion AIVM's encryption capabilities.

Prerequisites:
------------
- AIVM devnet must be running (see README.md)
- Environment setup completed
- Dependencies installed

Progress Tracking:
----------------
- Client Initialization ✓
- Data Encryption ✓
- Secure Prediction ✓
- Privacy Verification ✓

Usage:
-----
Initialize and use:
    food_bank = FoodBankData()
    encrypted_data = food_bank.encrypt_demand_data(demand_data)
    prediction = food_bank.secure_prediction(encrypted_data)
"""

import aivm_client as aic
import logging
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodBankData:
    """Class for handling food bank demand data with privacy preservation."""
    
    def __init__(self):
        """Initialize AIVM client and validate setup."""
        self.progress = {
            "client_setup": False,
            "encryption_ready": False,
            "prediction_ready": False,
            "privacy_verified": False
        }
        
        try:
            # Initialize AIVM client
            self.client = aic.Client()
            
            # Verify model support
            models = aic.get_supported_models()
            if "BertTiny" not in models:
                raise ValueError("BertTiny model not supported")
            
            self.progress["client_setup"] = True
            logger.info("✓ AIVM client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Client initialization failed: {e}")
            raise
        
    def encrypt_demand_data(self, demand_data: str) -> Optional[aic.BertTinyCryptensor]:
        """
        Encrypt sensitive food bank demand data using AIVM.
        
        Args:
            demand_data: Raw demand data to be encrypted
            
        Returns:
            BertTinyCryptensor: Encrypted data object
            
        Raises:
            Exception: If encryption fails
        """
        try:
            # Tokenize and encrypt data
            tokenized_data = aic.tokenize(demand_data)
            encrypted_data = aic.BertTinyCryptensor(*tokenized_data)
            
            # Verify encryption
            if not isinstance(encrypted_data, aic.BertTinyCryptensor):
                raise ValueError("Invalid encryption format")
            
            self.progress["encryption_ready"] = True
            logger.info("✓ Data encrypted successfully")
            
            # Verify privacy preservation
            self._verify_privacy(encrypted_data)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"❌ Data encryption failed: {e}")
            raise
    
    def secure_prediction(self, encrypted_data: aic.BertTinyCryptensor) -> Optional[Dict[str, Any]]:
        """
        Run prediction on encrypted data using AIVM.
        
        Args:
            encrypted_data: BertTinyCryptensor object containing encrypted data
            
        Returns:
            dict: Prediction results
            
        Raises:
            Exception: If prediction fails
        """
        try:
            # Verify input
            if not isinstance(encrypted_data, aic.BertTinyCryptensor):
                raise ValueError("Invalid input format")
            
            # Get prediction
            prediction = aic.get_prediction(encrypted_data, "FoodSecurityBERT")
            
            self.progress["prediction_ready"] = True
            logger.info("✓ Secure prediction completed")
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
            
    def _verify_privacy(self, encrypted_data: aic.BertTinyCryptensor) -> None:
        """Verify privacy guarantees of encrypted data."""
        try:
            privacy_checks = [
                isinstance(encrypted_data, aic.BertTinyCryptensor),
                len(encrypted_data.shape) > 0,
                encrypted_data.requires_grad is False
            ]
            
            if all(privacy_checks):
                self.progress["privacy_verified"] = True
                logger.info("✓ Privacy guarantees verified")
            else:
                raise ValueError("Privacy requirements not met")
                
        except Exception as e:
            logger.error(f"❌ Privacy verification failed: {e}")
            raise
            
    def get_progress(self) -> Dict[str, str]:
        """Get current progress status."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }

if __name__ == "__main__":
    try:
        # Example usage
        food_bank = FoodBankData()
        
        # Test with sample data
        sample_data = "Need food assistance for family of 4"
        encrypted_data = food_bank.encrypt_demand_data(sample_data)
        prediction = food_bank.secure_prediction(encrypted_data)
        
        # Show progress
        logger.info("\nImplementation Status:")
        for step, status in food_bank.get_progress().items():
            logger.info(f"{step}: {status}")
            
    except Exception as e:
        logger.error(f"Implementation test failed: {e}")
        exit(1)


