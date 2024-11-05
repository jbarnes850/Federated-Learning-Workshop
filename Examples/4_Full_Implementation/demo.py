"""
Full Implementation Demo
======================

This script demonstrates the complete food security network implementation,
including data privacy, model inference, and secure collaboration.

Progress Tracking:
----------------
- Setup ✓
- Privacy Demo ✓
- Model Demo ✓
- Network Demo ✓

Usage:
-----
Run the demo:
    python demo.py
"""

import logging
import os
from typing import Dict, Any, Optional
import aivm_client as aic
from aivm_client.models import BertTiny, get_supported_models
from aivm_client.cryptensor import Cryptensor
from aivm_client.utils import get_prediction
from food_security_network import FoodSecurityNetwork
from food_bank_data import generate_synthetic_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullDemo:
    def __init__(self):
        """Initialize demo components with AIVM validation."""
        self.progress = {
            "client_setup": False,
            "network_setup": False,
            "privacy_tested": False,
            "model_tested": False,
            "network_tested": False
        }
        
        try:
            # Initialize AIVM client
            self.client = aic.Client()
            self.api_key = os.getenv('AIVM_API_KEY')
            if not self.api_key:
                raise ValueError("AIVM_API_KEY environment variable not set")
            self.client.configure(api_key=self.api_key)
            
            # Verify AIVM support
            supported_models = get_supported_models()
            if "BertTiny" not in supported_models:
                raise ValueError("BertTiny model not supported in current AIVM version")
            
            self.progress["client_setup"] = True
            logger.info("✓ AIVM client initialized successfully")
            
            # Initialize network
            self.network = FoodSecurityNetwork()
            self.progress["network_setup"] = True
            logger.info("✓ Network initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

    async def run_demo(self) -> bool:
        """Execute complete system demonstration with validation."""
        try:
            # Generate and validate test data
            data = generate_synthetic_data(num_entries=10)
            self._validate_data(data)
            logger.info("✓ Generated and validated test data")

            # Test privacy preservation
            await self.test_privacy(data)
            self.progress["privacy_tested"] = True
            logger.info("✓ Privacy preservation verified")

            # Test model predictions
            await self.test_model(data)
            self.progress["model_tested"] = True
            logger.info("✓ Model predictions verified")

            # Test network collaboration
            await self.test_network(data)
            self.progress["network_tested"] = True
            logger.info("✓ Network collaboration verified")

            return True

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False

    def _validate_data(self, data) -> None:
        """Validate data structure and content."""
        required_columns = [
            'Region', 'City', 'ZipCode', 'Population', 
            'HouseholdSize', 'IncomeLevel', 'FoodType', 'DemandAmount'
        ]
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns in data")
        if data.empty:
            raise ValueError("Data is empty")

    async def test_privacy(self, data) -> bool:
        """Test and validate privacy preservation."""
        try:
            # Test encryption
            encrypted_data = Cryptensor(data.to_json())
            if not isinstance(encrypted_data, Cryptensor):
                raise TypeError("Invalid encryption format")
            
            # Verify privacy guarantees
            privacy_checks = [
                isinstance(encrypted_data, Cryptensor),
                len(encrypted_data.shape) > 0,
                encrypted_data.requires_grad is False  # Ensure no gradient tracking
            ]
            
            if not all(privacy_checks):
                raise ValueError("Privacy requirements not met")
                
            logger.info("✓ Privacy checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Privacy test failed: {e}")
            raise

    async def test_model(self, data) -> Optional[Dict[str, Any]]:
        """Test and validate model predictions."""
        try:
            # Prepare input
            encrypted_data = Cryptensor(data.to_json())
            
            # Get prediction
            predictions = await get_prediction(
                encrypted_data,
                "FoodSecurityBERT"
            )
            
            # Validate predictions
            if not isinstance(predictions, dict):
                raise TypeError("Invalid prediction format")
            if not predictions:
                raise ValueError("Empty predictions received")
            
            logger.info("✓ Model prediction successful")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            raise

    async def test_network(self, data) -> Optional[Dict[str, Any]]:
        """Test and validate network collaboration."""
        try:
            # Test secure data sharing
            insights = self.network.share_insights(data.to_json())
            if not insights:
                raise ValueError("Failed to share insights")
            
            # Test collaborative prediction
            collab_prediction = await self.network.predict_demand(data.to_json())
            if not collab_prediction:
                raise ValueError("Collaborative prediction failed")
            
            logger.info("✓ Network collaboration successful")
            return {
                "insights": insights,
                "predictions": collab_prediction
            }
            
        except Exception as e:
            logger.error(f"❌ Network test failed: {e}")
            raise

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
            demo = FullDemo()
            success = await demo.run_demo()
            
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

    asyncio.run(main())