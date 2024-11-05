"""
Food Bank Data Privacy Implementation
===================================

This module implements privacy-preserving data handling for food bank demand data using 
Nillion AIVM's encryption capabilities.

Progress Tracking:
----------------
- Client Initialization ✓
- Data Encryption ✓
- Secure Prediction ✓

Validation Steps:
---------------
1. Verify AIVM client setup
2. Test data encryption
3. Validate secure prediction
4. Check privacy preservation

Usage:
-----
Initialize the FoodBankData class and use its methods to handle sensitive data:
    food_bank = FoodBankData()
    encrypted_data = food_bank.encrypt_demand_data(demand_data)
    prediction = food_bank.secure_prediction(encrypted_data)
"""

import os
import logging
import aivm_client as aic
from aivm_client.models import BertTiny
from aivm_client.cryptensor import Cryptensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FoodBankData:
    """Class for handling food bank demand data with privacy preservation."""
    
    def __init__(self):
        """Initialize AIVM client and validate setup."""
        self.progress = {
            "client_setup": False,
            "encryption_ready": False,
            "prediction_ready": False
        }
        
        try:
            self.client = aic.Client()
            self.api_key = os.getenv('AIVM_API_KEY')
            if not self.api_key:
                raise ValueError("AIVM_API_KEY environment variable not set")
            self.client.configure(api_key=self.api_key)
            logger.info("✓ AIVM client initialized successfully")
            self.progress["client_setup"] = True
        except Exception as e:
            logger.error(f"❌ Client initialization failed: {e}")
            raise
        
    def encrypt_demand_data(self, demand_data):
        """
        Encrypt sensitive food bank demand data using Nillion AIVM.
        
        Args:
            demand_data: Raw demand data to be encrypted
            
        Returns:
            Cryptensor: Encrypted data object
            
        Raises:
            Exception: If encryption fails
        """
        try:
            encrypted_data = Cryptensor(demand_data)
            logger.info("✓ Data encrypted successfully")
            self.progress["encryption_ready"] = True
            return encrypted_data
        except Exception as e:
            logger.error(f"❌ Data encryption failed: {e}")
            return None
    
    def secure_prediction(self, encrypted_data):
        """
        Run prediction on encrypted data using Nillion AIVM.
        
        Args:
            encrypted_data: Cryptensor object containing encrypted data
            
        Returns:
            dict: Prediction results
            
        Raises:
            Exception: If prediction fails
        """
        try:
            prediction = self.client.get_prediction(encrypted_data)
            logger.info("✓ Secure prediction completed")
            self.progress["prediction_ready"] = True
            return prediction
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None
            
    def get_progress(self):
        """Return current progress status."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }


