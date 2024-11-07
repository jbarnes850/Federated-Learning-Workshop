"""
Model Training Script
===================

This module implements the training pipeline for the food security BertTiny model
using Nillion AIVM for privacy-preserving deployment.

The script handles:
- Loading and preprocessing food bank request data
- Initializing a BertTiny model for text classification 
- Training the model with privacy-preserving techniques
- Evaluating model performance and saving checkpoints
- Preparing the model for deployment on Nillion AIVM

The training pipeline includes:
- Custom FoodBankDataset class for efficient data loading
- Tokenization using BERT tokenizer
- Optimization with AdamW and learning rate scheduling
- Progress tracking and logging functionality
- Model evaluation on validation data

Environment variables (see .env.local):
- AIVM_DEVNET_HOST: Host for AIVM development network
- AIVM_DEVNET_PORT: Port for AIVM development network
- AIVM_DEVNET_ENABLED: Flag to enable/disable AIVM devnet
- AIVM_LOG_LEVEL: Logging verbosity level
"""

import sys
import os
# Add Examples directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aivm_client as aic
import logging
from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import time
from utils.progress import log_progress, ProgressStatus
from colorama import init, Fore, Style
from torch.optim.lr_scheduler import ReduceLROnPlateau
init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodBankDataset(Dataset):
    """Custom dataset for food bank data."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ModelTrainer:
    def __init__(self):
        """Initialize training components."""
        self.progress = {
            "setup_complete": False,
            "data_prepared": False,
            "model_trained": False,
            "model_deployed": False
        }
        
        try:
            # Initialize model components
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model_name = "prajjwal1/bert-tiny"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2  
            ).to(self.device)
            
            self.progress["setup_complete"] = True
            log_progress("Model Setup", ProgressStatus.COMPLETE)
            logger.info("✓ Training environment initialized")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            log_progress("Model Setup", ProgressStatus.FAILED)
            raise

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data."""
        try:
            # Load synthetic data
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "synthetic_data.csv"
            )
            
            # Check if file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Synthetic data not found at {data_path}. "
                    "Please run food_bank_data.py first."
                )
            
            # Load data
            try:
                data = pd.read_csv(data_path)
            except pd.errors.EmptyDataError:
                raise ValueError("Synthetic data file is empty")
            except Exception as e:
                raise ValueError(f"Error reading synthetic data: {str(e)}")
                
            # Validate required columns
            required_columns = [
                'FoodType', 'Population', 'HouseholdSize', 
                'IncomeLevel', 'DemandAmount', 'EmergencyStatus', 
                'SeasonalFactor'
            ]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Create text inputs matching the generated data format
            texts = data.apply(
                lambda row: (
                    f"Food Type: {row['FoodType']}, "
                    f"Location Type: {'Urban' if row['Population'] > 1000000 else 'Suburban' if row['Population'] > 500000 else 'Rural'}, "
                    f"Demographics: {row['Population']:,} residents in {row['HouseholdSize']}-person households, "
                    f"Economic Status: {row['IncomeLevel']} income area, "
                    f"Current Demand: {row['DemandAmount']}, "
                    f"Emergency Status: {row['EmergencyStatus']}, "
                    f"Season: {row['SeasonalFactor']}"
                ),
                axis=1
            )
            
            # Create balanced labels with better thresholding
            demand_mean = data['DemandAmount'].mean()
            demand_std = data['DemandAmount'].std()
            
            # Use mean + std as threshold for more balanced classes
            demand_threshold = demand_mean + (0.5 * demand_std)
            labels = (data['DemandAmount'] > demand_threshold).astype(int)
            
            # Verify class balance
            class_counts = labels.value_counts()
            if min(class_counts.values) < len(labels) * 0.2:  # Ensure at least 20% in minority class
                logger.warning(f"Imbalanced classes detected. Adjusting threshold...")
               
                demand_threshold = data['DemandAmount'].median()
                labels = (data['DemandAmount'] > demand_threshold).astype(int)
                class_counts = labels.value_counts()
            
            # Print data statistics
            logger.info(f"Data loaded successfully:")
            logger.info(f"Total samples: {len(data)}")
            logger.info(f"Demand threshold: {demand_threshold:.0f}")
            logger.info(f"Mean demand: {demand_mean:.0f}")
            logger.info(f"Std demand: {demand_std:.0f}")
            logger.info(f"Label distribution: {class_counts.to_dict()}")
            
            # Split into train and validation
            train_size = int(0.8 * len(texts))
            train_texts = texts[:train_size]
            train_labels = labels[:train_size]
            val_texts = texts[train_size:]
            val_labels = labels[train_size:]
            
            # Create datasets
            train_dataset = FoodBankDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = FoodBankDataset(val_texts, val_labels, self.tokenizer)
            
            # Calculate class weights
            class_counts = train_labels.value_counts()
            weights = torch.FloatTensor([
                1.0 / class_counts[0],
                1.0 / class_counts[1]
            ])
            weights = weights / weights.sum() 
            
            # Create weighted sampler
            sample_weights = torch.FloatTensor([
                weights[label] for label in train_labels
            ])
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            # Create dataloaders with weighted sampler
            train_loader = DataLoader(
                train_dataset, 
                batch_size=32,  
                sampler=sampler,
                num_workers=0  
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=16,
                num_workers=0  
            )
            
            self.progress["data_prepared"] = True
            log_progress("Data Preparation", ProgressStatus.COMPLETE)
            logger.info(f"�� Data preparation complete - {len(texts)} samples")
            logger.info(f"Class distribution - High Demand: {sum(train_labels)}, Low Demand: {len(train_labels)-sum(train_labels)}")
            
            return train_loader, val_loader
                
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {str(e)}")
            log_progress("Data Preparation", ProgressStatus.FAILED)
            raise

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 5) -> None:
        """Train model with improved learning process."""
        try:
            logger.info(f"\n{Fore.CYAN}Starting Model Training{Style.RESET_ALL}")
            
            # Initialize best state tracking
            best_val_accuracy = 0.0
            best_state = None
            patience = 3
            no_improve = 0
            
            
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=2e-5,          
                weight_decay=0.01  
            )
            
            
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=total_steps // 10,  
                num_training_steps=total_steps
            )
            
            for epoch in range(num_epochs):
                # Training phase
                self.model.train()
                total_train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    # Move data to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # Track metrics
                    total_train_loss += loss.item()
                    predictions = torch.argmax(outputs.logits, dim=1)
                    train_correct += (predictions == labels).sum().item()
                    train_total += labels.size(0)
                    
                    # Log progress
                    if batch_idx % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs} - "
                            f"Batch {batch_idx}/{len(train_loader)} - "
                            f"Loss: {loss.item():.4f} - "
                            f"LR: {current_lr:.2e}"
                        )
                
                # Calculate training metrics
                avg_train_loss = total_train_loss / len(train_loader)
                train_accuracy = train_correct / train_total
                
                # Validation phase
                val_loss, val_accuracy = self._validate_epoch(val_loader)
                
                # Log epoch results
                logger.info(f"\nEpoch {epoch + 1} Results:")
                logger.info(f"Training Loss: {avg_train_loss:.4f}")
                logger.info(f"Training Accuracy: {train_accuracy:.2%}")
                logger.info(f"Validation Loss: {val_loss:.4f}")
                logger.info(f"Validation Accuracy: {val_accuracy:.2%}")
                
                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_state = self.model.state_dict().copy()  
                    no_improve = 0
                    logger.info(f"{Fore.GREEN}✓ New best model saved!{Style.RESET_ALL}")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info("Early stopping triggered!")
                        break
            
            # Restore best model
            if best_state is not None:
                self.model.load_state_dict(best_state)
                logger.info(f"Restored best model with {best_val_accuracy:.2%} validation accuracy")
            
            self.progress["model_trained"] = True
            log_progress("Model Training", ProgressStatus.COMPLETE)
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            log_progress("Model Training", ProgressStatus.FAILED)
            raise

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model performance with proper metrics."""
        try:
            self.model.eval()
            total_val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Move data to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Calculate metrics
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                    # Log batch progress
                    if batch_idx % 5 == 0:
                        batch_accuracy = correct / total
                        logger.info(
                            f"Validation Batch {batch_idx}/{len(val_loader)} - "
                            f"Loss: {loss.item():.4f}, "
                            f"Accuracy: {batch_accuracy:.2%}"
                        )
            
            avg_val_loss = total_val_loss / len(val_loader)
            accuracy = correct / total
            
            return avg_val_loss, accuracy
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return float('inf'), 0.0

    def deploy_model(self, model_path: str = "models/food_security_bert.onnx") -> None:
        """Deploy model to AIVM."""
        try:
            logger.info(f"\n{Fore.CYAN}Model Deployment{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Export to ONNX
            dummy_input = self.tokenizer(
                "Example text",
                return_tensors="pt",
                padding="max_length",
                max_length=128,
                truncation=True
            )
            
            input_names = ["input_ids", "attention_mask"]
            dynamic_axes = {name: {0: "batch_size"} for name in input_names}
            dynamic_axes.update({"logits": {0: "batch_size"}})
            
            # Export model with versioned name
            timestamp = int(time.time())
            model_name = f"FoodSecurityBERT_{timestamp}"
            
            torch.onnx.export(
                self.model,
                tuple(dummy_input[k].to(self.device) for k in input_names),
                model_path,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=14
            )
            
            # Upload to AIVM with versioned name
            aic.upload_bert_tiny_model(model_path, model_name)
            
            self.progress["model_deployed"] = True
            log_progress("Model Deployment", ProgressStatus.COMPLETE)
            logger.info(f"{Fore.GREEN}✓ Model deployed successfully as {model_name}{Style.RESET_ALL}")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            log_progress("Model Deployment", ProgressStatus.FAILED)
            raise

    def get_progress(self) -> Dict[str, bool]:
        """Return current progress status."""
        return self.progress

if __name__ == "__main__":
    try:
        # Check if synthetic data exists
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "synthetic_data.csv"
        )
        if not os.path.exists(data_path):
            logger.error("❌ Synthetic data not found. Please run the following first:")
            logger.error("   python Examples/2_Data_Privacy/food_bank_data.py")
            exit(1)
            
        logger.info("Starting training pipeline...")
        log_progress("Training Pipeline", ProgressStatus.IN_PROGRESS)
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Prepare data
        train_loader, val_loader = trainer.prepare_data()
        
        # Show training time estimate
        logger.info("Note: Training will take approximately 5-10 minutes on CPU...")
        log_progress("Model Training", ProgressStatus.IN_PROGRESS)
        
        # Train model
        trainer.train_model(train_loader, val_loader)
        
        # Deploy model
        trainer.deploy_model()
        
        # Show final status
        logger.info("\nTraining Pipeline Status:")
        for step, status in trainer.get_progress().items():
            logger.info(f"{step}: {'✓' if status else '❌'}")
        
        log_progress("Training Pipeline", ProgressStatus.COMPLETE)
        logger.info("✓ Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        log_progress("Training Pipeline", ProgressStatus.FAILED)
        exit(1)