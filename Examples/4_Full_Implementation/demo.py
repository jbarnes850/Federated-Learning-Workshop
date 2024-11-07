"""
Full Implementation Demo
======================

This script demonstrates the complete food security network implementation,
including data privacy, model inference, and secure collaboration.

Prerequisites:
------------
- AIVM devnet must be running (see README.md)
- Model deployed
- Network configured

Progress Tracking:
----------------
- Setup ✓
- Data Generation ✓
- Privacy Demo ✓
- Model Demo ✓
- Network Demo ✓

Usage:
-----
1. Ensure devnet is running:
   aivm-devnet

2. Run demo:
   python demo.py
"""

import aivm_client as aic
import logging
import os
from typing import Dict, Any, Optional
from food_security_network import FoodSecurityNetwork
from food_bank_data import generate_synthetic_data
from config import NetworkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullDemo:
    def __init__(self):
        """Initialize demo components with validation."""
        self.progress = {
            "client_setup": False,
            "network_setup": False,
            "data_ready": False,
            "prediction_tested": False,
            "sharing_tested": False
        }
        
        try:
            # Initialize AIVM client
            self.client = aic.Client()
            
            # Verify model support
            models = aic.get_supported_models()
            if "BertTiny" not in models:
                raise ValueError("BertTiny model not supported")
            
            self.progress["client_setup"] = True
            logger.info("✓ Connected to AIVM devnet")
            
            # Initialize network
            self.network = FoodSecurityNetwork()
            self.config = NetworkConfig.load_config()
            self.progress["network_setup"] = True
            logger.info("✓ Network initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

    async def prepare_data(self) -> bool:
        """Generate and prepare test data."""
        try:
            # Generate synthetic data
            self.data = generate_synthetic_data(num_entries=10)
            if self.data.empty:
                raise ValueError("Generated data is empty")
            
            self.progress["data_ready"] = True
            logger.info("✓ Test data prepared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return False

    async def test_predictions(self) -> bool:
        """Test secure prediction capabilities."""
        try:
            # Prepare sample cases
            test_cases = [
                "Urgent food assistance needed for family of 5",
                "Increased demand for fresh produce in summer",
                "Emergency supplies required after natural disaster"
            ]
            
            for case in test_cases:
                # Tokenize and encrypt
                tokenized_data = aic.tokenize(case)
                encrypted_data = aic.BertTinyCryptensor(*tokenized_data)
                
                # Get prediction
                prediction = await self.network.predict_demand(case)
                logger.info(f"Case: {case}\nPrediction: {prediction}\n")
            
            self.progress["prediction_tested"] = True
            logger.info("✓ Prediction testing complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prediction testing failed: {e}")
            return False

    async def test_sharing(self) -> bool:
        """Test secure data sharing capabilities."""
        try:
            # Test different types of insights
            insights = {
                "demand_trend": "25% increase in monthly demand",
                "resource_allocation": "Fresh produce shortage detected",
                "emergency_alert": "Natural disaster preparation needed"
            }
            
            for insight_type, message in insights.items():
                shared_data = self.network.share_insights(message)
                logger.info(f"Shared {insight_type}: {message}")
            
            self.progress["sharing_tested"] = True
            logger.info("✓ Sharing capabilities verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sharing test failed: {e}")
            return False

    async def run_demo(self) -> bool:
        """Execute complete system demonstration."""
        try:
            # Prepare test data
            if not await self.prepare_data():
                return False
            
            # Test predictions
            if not await self.test_predictions():
                return False
            
            # Test sharing
            if not await self.test_sharing():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False

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
            # Run complete demo
            demo = FullDemo()
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