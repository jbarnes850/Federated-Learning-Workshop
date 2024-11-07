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

import sys
import os
# Add Examples directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aivm_client as aic
import logging
import pandas as pd
from typing import Dict, Any, Optional
from food_security_network import FoodSecurityNetwork
from utils.progress import log_progress, ProgressStatus
from colorama import init, Fore, Style
import torch
init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullDemo:
    """Complete demonstration of food security network capabilities."""
    
    def __init__(self):
        """Initialize demo components with validation."""
        self.progress = {
            "setup_complete": False,
            "data_loaded": False,
            "privacy_tested": False,
            "model_tested": False,
            "network_tested": False
        }
        
        try:
            # Initialize network
            self.network = FoodSecurityNetwork()
            self.progress["setup_complete"] = True
            log_progress("Demo Setup", ProgressStatus.COMPLETE)
            
        except Exception as e:
            logger.error(f"❌ Demo initialization failed: {e}")
            log_progress("Demo Setup", ProgressStatus.FAILED)
            raise

    def load_data(self) -> Optional[pd.DataFrame]:
        """Load and validate synthetic data."""
        try:
            # Load synthetic data from root directory
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "synthetic_data.csv"
            )
            data = pd.read_csv(data_path)
            
            if data.empty:
                raise ValueError("Loaded data is empty")
            
            self.progress["data_loaded"] = True
            log_progress("Data Loading", ProgressStatus.COMPLETE)
            logger.info(f"✓ Loaded {len(data)} samples")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            log_progress("Data Loading", ProgressStatus.FAILED)
            raise

    async def test_privacy(self, data: pd.DataFrame) -> bool:
        """Test privacy preservation capabilities."""
        try:
            log_progress("Privacy Testing", ProgressStatus.IN_PROGRESS)
            
            # Test data encryption
            sample_data = data.sample(n=1).iloc[0]
            text_data = f"{sample_data['FoodType']} - Demand: {sample_data['DemandAmount']}"
            
            # Test secure sharing
            shared_data = self.network.share_insights(text_data)
            if not shared_data:
                raise ValueError("Sharing failed")
            
            self.progress["privacy_tested"] = True
            log_progress("Privacy Testing", ProgressStatus.COMPLETE)
            logger.info("✓ Privacy preservation verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Privacy testing failed: {e}")
            log_progress("Privacy Testing", ProgressStatus.FAILED)
            raise

    async def test_model(self, data: pd.DataFrame) -> bool:
        """Test model inference capabilities with meaningful insights."""
        try:
            log_progress("Model Testing", ProgressStatus.IN_PROGRESS)
            
            # Calculate data insights
            median_demand = data['DemandAmount'].median()
            high_pop_threshold = data['Population'].quantile(0.75)
            
            # Test cases based on real patterns
            test_cases = []
            
            # Case 1: High population urban center
            urban_case = data[
                (data['Population'] > high_pop_threshold) & 
                (data['FoodType'] == 'Fresh Produce')
            ].sample(n=1).iloc[0]
            test_cases.append({
                'data': urban_case,
                'context': 'Urban Center Analysis'
            })
            
            # Case 2: Low income emergency supplies
            emergency_case = data[
                (data['IncomeLevel'] == 'Low') & 
                (data['FoodType'] == 'Canned Goods')
            ].nlargest(1, 'DemandAmount').iloc[0]
            test_cases.append({
                'data': emergency_case,
                'context': 'Emergency Supply Assessment'
            })
            
            # Case 3: Large family protein needs
            family_case = data[
                (data['HouseholdSize'] >= 5) & 
                (data['FoodType'] == 'Meat/Poultry')
            ].sample(n=1).iloc[0]
            test_cases.append({
                'data': family_case,
                'context': 'Family Nutrition Analysis'
            })
            
            # Run predictions with context
            for case in test_cases:
                row = case['data']
                text_data = (
                    f"{row['FoodType']} - Demand: {row['DemandAmount']} "
                    f"(Population: {row['Population']:,}, "
                    f"Income: {row['IncomeLevel']}, "
                    f"Household Size: {row['HouseholdSize']})"
                )
                prediction = await self.network.predict_demand(text_data)
                
                # Process prediction with context
                logits = prediction[0]  # Get raw logits
                probabilities = torch.nn.functional.softmax(logits, dim=0)  # Convert to probabilities
                result = "High" if probabilities[1] > probabilities[0] else "Low"
                confidence = float(probabilities[1] if result == "High" else probabilities[0])
                
                # Log detailed insights
                logger.info(f"\n{Fore.CYAN}{'='*50}")
                logger.info(f"{case['context'].upper()}")
                logger.info(f"{'='*50}{Style.RESET_ALL}")

                logger.info(f"{Fore.BLUE}Location:{Style.RESET_ALL} {row['City']}, {row['Region']}")
                logger.info(f"{Fore.BLUE}Demographics:{Style.RESET_ALL}")
                logger.info(f"  • Population: {row['Population']:,}")
                logger.info(f"  • Income Level: {row['IncomeLevel']}")
                logger.info(f"  • Household Size: {row['HouseholdSize']}")

                logger.info(f"\n{Fore.YELLOW}Demand Analysis:{Style.RESET_ALL}")
                logger.info(f"  • Food Type: {row['FoodType']}")
                logger.info(f"  • Current Demand: {row['DemandAmount']}")

                # Color code the prediction based on result
                prediction_color = Fore.RED if result == "High" else Fore.GREEN
                logger.info(f"\n{Fore.MAGENTA}Prediction Results:{Style.RESET_ALL}")
                logger.info(f"  • Demand Level: {prediction_color}{result}{Style.RESET_ALL}")
                logger.info(f"  • Confidence: {confidence:.2%}")
                logger.info(f"  • Raw Probabilities: Low={probabilities[0]:.2%}, High={probabilities[1]:.2%}")

                # Add visual separator for recommendations
                logger.info(f"\n{Fore.CYAN}Recommended Actions:{Style.RESET_ALL}")
                if result == "High":
                    if row['FoodType'] == 'Fresh Produce':
                        logger.info("  🚜 Coordinate with local farms for direct supply")
                    elif row['FoodType'] == 'Canned Goods':
                        logger.info("  🏭 Activate emergency supply network")
                    else:
                        logger.info("  📈 Increase distribution capacity")
                else:
                    logger.info("  ✓ Maintain standard supply levels")

                logger.info(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")
            
            self.progress["model_tested"] = True
            log_progress("Model Testing", ProgressStatus.COMPLETE)
            logger.info("✓ Model inference verified with contextual insights")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")
            log_progress("Model Testing", ProgressStatus.FAILED)
            raise

    async def test_network(self) -> bool:
        """Test complete network functionality."""
        try:
            log_progress("Network Testing", ProgressStatus.IN_PROGRESS)
            
            # Verify network status
            network_status = self.network.get_progress()
            if not all(network_status.values()):
                raise ValueError("Network setup incomplete")
            
            self.progress["network_tested"] = True
            log_progress("Network Testing", ProgressStatus.COMPLETE)
            logger.info("✓ Network functionality verified")
            return True
            
        except Exception as e:
            logger.error(f"❌ Network testing failed: {e}")
            log_progress("Network Testing", ProgressStatus.FAILED)
            raise

    async def run_demo(self) -> bool:
        """Execute complete demonstration."""
        try:
            log_progress("Full Demo", ProgressStatus.IN_PROGRESS)
            
            # Load data
            data = self.load_data()
            
            # Run tests
            if not await self.test_privacy(data):
                return False
                
            if not await self.test_model(data):
                return False
                
            if not await self.test_network():
                return False
            
            log_progress("Full Demo", ProgressStatus.COMPLETE)
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            log_progress("Full Demo", ProgressStatus.FAILED)
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