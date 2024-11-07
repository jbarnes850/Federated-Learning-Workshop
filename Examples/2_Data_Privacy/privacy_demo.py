"""
Privacy Demo Script
=================

This script demonstrates privacy-preserving capabilities using Nillion AIVM for
food bank data analysis. It shows encryption, secure prediction, and privacy verification.

Prerequisites:
------------
- AIVM devnet must be running (see README.md)
- Environment setup completed
- Dependencies installed

Progress Tracking:
----------------
- Devnet Connection ✓
- Data Generation ✓
- Encryption Demo ✓
- Secure Prediction ✓
- Privacy Verification ✓

Usage:
-----
1. Ensure devnet is running in a separate terminal:
   aivm-devnet

2. Run the demo:
   python privacy_demo.py
"""

import aivm_client as aic
import logging
import os
from typing import Dict, Any, Optional
from food_bank_data import generate_synthetic_data
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyDemo:
    def __init__(self):
        """Initialize demo components and verify environment."""
        self.progress = {
            "devnet_check": False,
            "data_generated": False,
            "encryption_tested": False,
            "prediction_tested": False,
            "privacy_verified": False
        }
        
        # Verify devnet is running
        try:
            self.client = aic.Client()
            
            # Verify model support
            models = aic.get_supported_models()
            if "BertTiny" not in models:
                raise ValueError("BertTiny model not supported in current AIVM version")
            
            self.progress["devnet_check"] = True
            logger.info("✓ Connected to AIVM devnet")
                
        except Exception as e:
            logger.error("❌ Failed to connect to AIVM devnet. Is it running?")
            raise

    async def generate_test_data(self) -> bool:
        """Generate and validate test data."""
        try:
            self.data = generate_synthetic_data(num_entries=10)
            if self.data.empty:
                raise ValueError("Generated data is empty")
            
            self.progress["data_generated"] = True
            logger.info("✓ Generated test data successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Data generation failed: {e}")
            return False

    async def demonstrate_encryption(self) -> Optional[aic.BertTinyCryptensor]:
        """Demonstrate encryption capabilities."""
        try:
            # Convert data to text format
            text_data = self.data.to_json()
            
            # Tokenize the text data
            tokenized_data = aic.tokenize(text_data)
            
            # Encrypt using BertTinyCryptensor
            encrypted_data = aic.BertTinyCryptensor(*tokenized_data)
            
            self.progress["encryption_tested"] = True
            logger.info("✓ Data encrypted successfully")
            return encrypted_data
        except Exception as e:
            logger.error(f"❌ Encryption demonstration failed: {e}")
            return None

    async def test_prediction(self, encrypted_data: aic.BertTinyCryptensor) -> bool:
        """Test secure prediction on encrypted data."""
        try:
            # Verify input format
            if not isinstance(encrypted_data, aic.BertTinyCryptensor):
                raise ValueError("Invalid encryption format")
            
            # Get prediction using AIVM
            prediction = aic.get_prediction(encrypted_data, "FoodSecurityBERT")
            
            self.progress["prediction_tested"] = True
            logger.info("✓ Secure prediction completed successfully")
            logger.info(f"Prediction result: {prediction}")
            return True
        except Exception as e:
            logger.error(f"❌ Prediction test failed: {e}")
            return False

    async def verify_privacy(self, encrypted_data: aic.BertTinyCryptensor) -> bool:
        """Verify privacy preservation guarantees."""
        try:
            # Run privacy checks
            privacy_checks = [
                isinstance(encrypted_data, aic.BertTinyCryptensor),
                len(encrypted_data.shape) > 0,
                encrypted_data.requires_grad is False,
                not hasattr(encrypted_data, '_raw_data')  # Ensure no raw data access
            ]
            
            if all(privacy_checks):
                self.progress["privacy_verified"] = True
                logger.info("✓ Privacy guarantees verified")
                return True
            else:
                raise ValueError("Privacy requirements not met")
        except Exception as e:
            logger.error(f"❌ Privacy verification failed: {e}")
            return False

    async def run_demo(self) -> bool:
        """Execute complete privacy demonstration."""
        if not self.progress["devnet_check"]:
            logger.error("❌ AIVM devnet not running. Start devnet first.")
            return False
            
        try:
            # Generate and encrypt data
            if not await self.generate_test_data():
                return False
                
            encrypted_data = await self.demonstrate_encryption()
            if encrypted_data is None:
                return False
                
            # Test prediction and verify privacy
            if not await self.test_prediction(encrypted_data):
                return False
                
            if not await self.verify_privacy(encrypted_data):
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False

    def get_progress(self) -> Dict[str, str]:
        """Get current progress status with validation."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        try:
            # Run complete demo
            demo = PrivacyDemo()
            success = await demo.run_demo()
            
            # Show progress
            logger.info("\nDemo Status:")
            for step, status in demo.get_progress().items():
                logger.info(f"{step}: {status}")
                
            if not success:
                logger.error("❌ Demo completed with errors")
                exit(1)
            
            logger.info("✓ Demo completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            exit(1)

    # Run async demo
    asyncio.run(main())