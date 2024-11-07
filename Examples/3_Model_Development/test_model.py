"""
Model Testing Script
==================

This module provides testing capabilities for the trained food security model,
ensuring proper inference and privacy preservation using Nillion AIVM.

Prerequisites:
------------
- AIVM devnet must be running (see README.md)
- Environment setup completed
- Model training completed (run train_model.py first)
- Dependencies installed

Progress Tracking:
----------------
- Setup Complete ✓
- Model Loading ✓
- Test Data Preparation ✓
- Inference Testing ✓

Features:
--------
- Automatic latest model detection
- Random test sample generation
- Privacy-preserving inference
- Confidence scoring
- Progress tracking

Usage:
-----
1. Ensure devnet is running:
   aivm-devnet

2. Run tests:
   python test_model.py

Note: This script automatically detects and uses the latest trained model
(FoodSecurityBERT_{timestamp}) from the AIVM network.
"""

import sys
import os
# Add Examples directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aivm_client as aic
import logging
import torch
from typing import Dict, Any, Optional, List
import pandas as pd
from utils.progress import log_progress, ProgressStatus
from colorama import init, Fore, Style
init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self):
        """Initialize testing components."""
        self.progress = {
            "setup_complete": False,
            "model_loaded": False,
            "test_data_prepared": False,
            "inference_tested": False
        }
        
        try:
            # Get available models
            self.available_models = aic.get_supported_models()
            logger.info(f"{Fore.BLUE}Available models: {self.available_models}{Style.RESET_ALL}")
            
            # Find the latest FoodSecurityBERT model
            food_security_models = [
                model for model in self.available_models 
                if model.startswith("FoodSecurityBERT_")
            ]
            
            if not food_security_models:
                raise ValueError("No FoodSecurityBERT models found. Please run training first.")
            
            # Sort by timestamp and get the latest
            self.model_name = sorted(food_security_models)[-1]
            logger.info(f"{Fore.GREEN}Using latest model: {self.model_name}{Style.RESET_ALL}")
            
            self.progress["model_loaded"] = True
            
            self.progress["setup_complete"] = True
            log_progress("Test Setup", ProgressStatus.COMPLETE)
            
        except Exception as e:
            logger.error(f"❌ Test setup failed: {e}")
            log_progress("Test Setup", ProgressStatus.FAILED)
            raise

    def prepare_test_data(self) -> Optional[List[str]]:
        """Prepare test data from synthetic dataset with meaningful insights."""
        try:
            # Load synthetic data
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "synthetic_data.csv"
            )
            data = pd.read_csv(data_path)
            
            # Calculate meaningful thresholds
            median_demand = data['DemandAmount'].median()
            high_population = data['Population'].quantile(0.75)
            
            # Create insightful test cases
            test_cases = []
            
            # High population, high demand area
            high_pop_area = data[data['Population'] > high_population].sample(n=1).iloc[0]
            test_cases.append(
                f"{high_pop_area['FoodType']} - Demand: {high_pop_area['DemandAmount']} "
                f"(High population area: {high_pop_area['City']})"
            )
            
            # Low income, high household size
            low_income_case = data[
                (data['IncomeLevel'] == 'Low') & 
                (data['HouseholdSize'] >= 5)
            ].sample(n=1).iloc[0]
            test_cases.append(
                f"{low_income_case['FoodType']} - Demand: {low_income_case['DemandAmount']} "
                f"(Low income, large household)"
            )
            
            # Fresh produce in high demand
            produce_case = data[
                data['FoodType'] == 'Fresh Produce'
            ].nlargest(1, 'DemandAmount').iloc[0]
            test_cases.append(
                f"Fresh Produce - Demand: {produce_case['DemandAmount']} "
                f"(Peak produce demand)"
            )
            
            # Emergency supplies (Canned Goods)
            emergency_case = data[
                data['FoodType'] == 'Canned Goods'
            ].nlargest(1, 'DemandAmount').iloc[0]
            test_cases.append(
                f"Canned Goods - Demand: {emergency_case['DemandAmount']} "
                f"(Emergency supplies)"
            )
            
            # Protein needs (Meat/Poultry)
            protein_case = data[
                (data['FoodType'] == 'Meat/Poultry') & 
                (data['HouseholdSize'] >= 4)
            ].sample(n=1).iloc[0]
            test_cases.append(
                f"Meat/Poultry - Demand: {protein_case['DemandAmount']} "
                f"(Family protein needs)"
            )
            
            self.progress["test_data_prepared"] = True
            log_progress("Data Preparation", ProgressStatus.COMPLETE)
            logger.info(f"✓ Prepared {len(test_cases)} realistic test scenarios")
            
            return test_cases
            
        except Exception as e:
            logger.error(f"❌ Test data preparation failed: {e}")
            log_progress("Data Preparation", ProgressStatus.FAILED)
            raise

    def run_inference_tests(self, test_texts: List[str]) -> bool:
        """Test model inference capabilities with contextual insights."""
        try:
            logger.info(f"\n{Fore.CYAN}Starting Inference Tests{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            log_progress("Model Testing", ProgressStatus.IN_PROGRESS)
            
            total_confidence = 0
            high_demand_count = 0
            
            for idx, text in enumerate(test_texts):
                # Tokenize and encrypt
                tokenized = aic.tokenize(text)
                encrypted_input = aic.BertTinyCryptensor(*tokenized)
                
                # Get prediction
                prediction = aic.get_prediction(encrypted_input, self.model_name)
                
                # Process prediction with proper probability calculation
                logits = prediction[0]  # Get raw logits
                probabilities = torch.nn.functional.softmax(logits, dim=0)  # Convert to probabilities
                result = 1 if probabilities[1] > probabilities[0] else 0
                confidence = float(probabilities[1] if result == 1 else probabilities[0])
                
                # Track metrics
                total_confidence += confidence
                if result == 1:
                    high_demand_count += 1
                
                # Log detailed insights
                logger.info(f"\n{Fore.YELLOW}Scenario {idx + 1}:{Style.RESET_ALL}")
                logger.info(f"{Fore.BLUE}Context:{Style.RESET_ALL} {text}")
                
                # Color code prediction based on result
                prediction_color = Fore.RED if result == 1 else Fore.GREEN
                prediction_text = 'High Demand Alert' if result == 1 else 'Normal Demand'
                logger.info(f"{Fore.BLUE}Prediction:{Style.RESET_ALL} {prediction_color}{prediction_text}{Style.RESET_ALL}")
                logger.info(f"{Fore.BLUE}Confidence:{Style.RESET_ALL} {confidence:.2%}")
                logger.info(f"{Fore.BLUE}Raw Probabilities:{Style.RESET_ALL} Low={probabilities[0]:.2%}, High={probabilities[1]:.2%}")
                
                # Color code recommendations
                logger.info(f"{Fore.MAGENTA}Recommendation:{Style.RESET_ALL} {
                    'Consider immediate resource allocation' if result == 1 
                    else 'Maintain standard supply levels'
                }")
            
            # Log aggregate insights
            logger.info(f"\n{Fore.CYAN}Aggregate Analysis:{Style.RESET_ALL}")
            logger.info(f"High Demand Scenarios: {high_demand_count}/{len(test_texts)}")
            logger.info(f"Average Confidence: {(total_confidence/len(test_texts)):.2%}")
            
            self.progress["inference_tested"] = True
            log_progress("Model Testing", ProgressStatus.COMPLETE)
            logger.info("✓ All inference tests completed with detailed insights")
            return True
            
        except Exception as e:
            logger.error(f"❌ Inference testing failed: {e}")
            log_progress("Model Testing", ProgressStatus.FAILED)
            raise

    def get_progress(self) -> Dict[str, bool]:
        """Return current progress status."""
        return self.progress

if __name__ == "__main__":
    try:
        logger.info("Starting model testing...")
        log_progress("Testing Pipeline", ProgressStatus.IN_PROGRESS)
        
        # Initialize tester
        tester = ModelTester()
        
        # Prepare test data
        test_texts = tester.prepare_test_data()
        
        # Run inference tests
        success = tester.run_inference_tests(test_texts)
        
        # Show final status
        logger.info("\nTesting Pipeline Status:")
        for step, status in tester.get_progress().items():
            logger.info(f"{step}: {'✓' if status else '❌'}")
        
        if success:
            log_progress("Testing Pipeline", ProgressStatus.COMPLETE)
            logger.info("✓ Testing pipeline completed successfully")
        else:
            log_progress("Testing Pipeline", ProgressStatus.FAILED)
            logger.error("❌ Testing pipeline completed with errors")
            exit(1)
            
    except Exception as e:
        logger.error(f"❌ Testing pipeline failed: {e}")
        log_progress("Testing Pipeline", ProgressStatus.FAILED)
        exit(1)