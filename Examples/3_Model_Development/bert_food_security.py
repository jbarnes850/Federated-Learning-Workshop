"""
Food Security BertTiny Model Implementation
=========================================

This module implements a custom BertTiny model for analyzing food security issues.
It leverages AIVM for model deployment and inference.

Prerequisites:
------------
- AIVM devnet must be running (see README.md)
- Environment setup completed
- Dependencies installed

Progress Tracking:
----------------
- Model Initialization ✓
- Model Loading ✓
- Model Deployment ✓

Usage:
-----
1. Ensure devnet is running:
   aivm-devnet

2. Initialize and deploy model:
   model = FoodSecurityBertTiny()
   model.deploy_model('path/to/model')
"""

import aivm_client as aic
import logging
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Any, Optional
import os
from colorama import init, Fore, Style
import time
init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodSecurityBertTiny(nn.Module):
    """Custom BertTiny implementation for food security analysis."""
    
    def __init__(self, num_labels: int = 2):
        """Initialize model components."""
        super(FoodSecurityBertTiny, self).__init__()
        
        self.progress = {
            "model_init": False,
            "bert_loaded": False,
            "deployment_ready": False
        }
        
        try:
            # Initialize BERT components
            self.model_name = "prajjwal1/bert-tiny"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.progress["model_init"] = True
            self.progress["bert_loaded"] = True
            logger.info(f"{Fore.GREEN}✓ Model initialized successfully{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Model initialization failed: {e}{Style.RESET_ALL}")
            raise

    def deploy_model(self, model_path: str) -> bool:
        """
        Deploy model to AIVM network.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"\n{Fore.CYAN}Model Deployment{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
            # Ensure the directory exists
            directory = os.path.dirname(model_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"✓ Created directory: {directory}")

            # Create unique model name with timestamp
            timestamp = int(time.time())
            model_name = f"FoodSecurityBERT_{timestamp}"

            # Export model to ONNX format
            dummy_input = self.tokenizer(
                "Example text",
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True
            )
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                model_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "logits": {0: "batch_size"}
                },
                opset_version=14
            )
            
            # Upload to AIVM with timestamped name
            aic.upload_bert_tiny_model(model_path, model_name)
            
            self.progress["deployment_ready"] = True
            logger.info(f"{Fore.GREEN}✓ Model deployed successfully{Style.RESET_ALL}")
            logger.info(f"{Fore.BLUE}Model Path: {model_path}{Style.RESET_ALL}")
            logger.info(f"{Fore.BLUE}Model Name: {model_name}{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            logger.error(f"{Fore.RED}❌ Deployment failed: {e}{Style.RESET_ALL}")
            return False

    def get_progress(self) -> Dict[str, str]:
        """Get current progress status."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }

if __name__ == "__main__":
    try:
        # Initialize model
        model = FoodSecurityBertTiny()
        
        # Test deployment
        success = model.deploy_model("models/food_security_bert.onnx")
        
        # Show progress
        logger.info(f"\n{Fore.CYAN}Model Status:{Style.RESET_ALL}")
        for step, status in model.get_progress().items():
            status_color = Fore.GREEN if status else Fore.RED
            status_symbol = "✓" if status else "❌"
            logger.info(f"{status_color}{step}: {status_symbol}{Style.RESET_ALL}")
            
        if not success:
            logger.error(f"{Fore.RED}❌ Model setup failed{Style.RESET_ALL}")
            exit(1)
            
        logger.info(f"{Fore.GREEN}✓ Model setup completed successfully{Style.RESET_ALL}")
        
    except Exception as e:
        logger.error(f"{Fore.RED}❌ Setup failed: {e}{Style.RESET_ALL}")
        exit(1)


