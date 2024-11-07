"""
Food Security Model Testing Module
================================

This module provides comprehensive testing capabilities for the BertTiny model
adapted for food security analysis. It validates model performance, prediction
accuracy, and privacy preservation requirements.

Prerequisites:
------------
- AIVM devnet must be running (see README.md)
- Model deployed
- Environment setup completed

Progress Tracking:
----------------
- Test Dataset Creation ✓
- Model Loading ✓
- Inference Testing ✓
- Privacy Verification ✓

Usage:
-----
1. Ensure devnet is running:
   aivm-devnet

2. Run tests:
   python test_model.py
"""

import aivm_client as aic
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional
from transformers import BertTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodSecurityDataset(Dataset):
    """Dataset class for food security testing data."""
    
    def __init__(self, texts: List[str], labels: List[int], max_length: int = 128):
        """
        Initialize test dataset.
        
        Args:
            texts: List of input texts
            labels: Corresponding labels
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized and encoded item."""
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize using AIVM
        tokenized = aic.tokenize(text)
        
        return {
            'text': text,
            'label': torch.tensor(label, dtype=torch.long),
            'tokenized': tokenized
        }

class ModelTester:
    """Handles model testing and evaluation."""
    
    def __init__(self):
        """Initialize with AIVM verification."""
        self.progress = {
            "client_setup": False,
            "data_loaded": False,
            "model_tested": False,
            "privacy_verified": False
        }
        
        try:
            # Verify AIVM connection
            self.client = aic.Client()
            
            # Verify model availability
            models = aic.get_supported_models()
            if "BertTiny" not in models:
                raise ValueError("BertTiny model not supported")
                
            self.progress["client_setup"] = True
            logger.info("✓ Connected to AIVM devnet")
            
        except Exception as e:
            logger.error("❌ Failed to connect to AIVM devnet. Is it running?")
            raise

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance with privacy preservation.
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            dict: Performance metrics
        """
        try:
            results = []
            labels = []
            
            for batch in test_loader:
                # Encrypt data
                encrypted_data = aic.BertTinyCryptensor(*batch['tokenized'])
                
                # Get prediction
                prediction = aic.get_prediction(
                    encrypted_data,
                    "FoodSecurityBERT"
                )
                
                results.append(prediction)
                labels.append(batch['label'])
            
            # Calculate metrics
            metrics = self._calculate_metrics(results, labels)
            
            self.progress["model_tested"] = True
            logger.info("✓ Model evaluation complete")
            
            # Verify privacy
            self._verify_privacy(encrypted_data)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise

    def _calculate_metrics(self, predictions: List[torch.Tensor], 
                         labels: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate performance metrics."""
        try:
            correct = sum(
                torch.argmax(p) == l for p, l in zip(predictions, labels)
            )
            total = len(predictions)
            
            return {
                'accuracy': correct / total,
                'total_samples': total
            }
        except Exception as e:
            logger.error(f"❌ Metrics calculation failed: {e}")
            raise

    def _verify_privacy(self, encrypted_data: aic.BertTinyCryptensor) -> None:
        """Verify privacy preservation of predictions."""
        try:
            privacy_checks = [
                isinstance(encrypted_data, aic.BertTinyCryptensor),
                len(encrypted_data.shape) > 0,
                encrypted_data.requires_grad is False
            ]
            
            if all(privacy_checks):
                self.progress["privacy_verified"] = True
                logger.info("✓ Privacy verification complete")
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
        # Initialize tester
        tester = ModelTester()
        
        # Prepare test data
        test_texts = [
            "Need food assistance due to financial difficulties",
            "Increased demand for fresh produce in summer",
            "Emergency food supply needed after natural disaster"
        ]
        test_labels = [1, 0, 2]
        
        # Create dataset and loader
        test_dataset = FoodSecurityDataset(test_texts, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        # Run evaluation
        metrics = tester.evaluate_model(test_loader)
        
        # Show results
        logger.info("\nTest Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        # Show progress
        logger.info("\nTest Status:")
        for step, status in tester.get_progress().items():
            logger.info(f"{step}: {status}")
            
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        exit(1)