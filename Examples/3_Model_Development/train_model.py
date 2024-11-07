"""
Model Training Script
===================

This module implements the training pipeline for the food security BertTiny model
using Nillion AIVM for privacy-preserving deployment.
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
                num_labels=2  # Binary classification
            ).to(self.device)
            
            self.progress["setup_complete"] = True
            log_progress("Model Setup", ProgressStatus.COMPLETE)
            logger.info("✓ Training environment initialized")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            log_progress("Model Setup", ProgressStatus.FAILED)
            raise

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data with enhanced context."""
        try:
            # Load synthetic data
            data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "synthetic_data.csv"
            )
            data = pd.read_csv(data_path)
            
            # Create enhanced text inputs with context
            texts = data.apply(
                lambda row: (
                    f"{row['FoodType']} - "
                    f"Population: {row['Population']:,}, "
                    f"Household: {row['HouseholdSize']}, "
                    f"Income: {row['IncomeLevel']}, "
                    f"Demand: {row['DemandAmount']}"
                ),
                axis=1
            )
            
            # Create balanced labels
            median_demand = data['DemandAmount'].median()
            labels = (data['DemandAmount'] > median_demand).astype(int)
            
            # Calculate class weights for balanced training
            class_counts = labels.value_counts()
            total_samples = len(labels)
            class_weights = {
                0: total_samples / (2 * class_counts[0]),
                1: total_samples / (2 * class_counts[1])
            }
            
            # Convert to tensor weights
            sample_weights = torch.FloatTensor([
                class_weights[label] for label in labels
            ])
            
            # Create weighted sampler for balanced batches
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            # Split into train and validation
            train_size = int(0.8 * len(texts))
            train_texts = texts[:train_size]
            train_labels = labels[:train_size]
            val_texts = texts[train_size:]
            val_labels = labels[train_size:]
            
            # Create datasets
            train_dataset = FoodBankDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = FoodBankDataset(val_texts, val_labels, self.tokenizer)
            
            # Create dataloaders with weighted sampler for training
            train_loader = DataLoader(
                train_dataset, 
                batch_size=32,
                sampler=sampler
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=32
            )
            
            self.progress["data_prepared"] = True
            log_progress("Data Preparation", ProgressStatus.COMPLETE)
            logger.info(f"✓ Data preparation complete - {len(texts)} samples")
            logger.info(f"Class distribution - High Demand: {class_counts[1]}, Low Demand: {class_counts[0]}")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            log_progress("Data Preparation", ProgressStatus.FAILED)
            raise

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 2) -> None:
        """Train model with validation."""
        try:
            logger.info(f"\n{Fore.CYAN}Starting Model Training{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
            # Use torch.optim.AdamW
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
            
            # Add warmup scheduler
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

            for epoch in range(num_epochs):
                logger.info(f"\n{Fore.YELLOW}Epoch {epoch + 1}/{num_epochs}{Style.RESET_ALL}")
                logger.info(f"{Fore.CYAN}{'='*30}{Style.RESET_ALL}")
                
                start_time = time.time()
                self.model.train()
                total_train_loss = 0
                right_predictions = 0

                try:
                    for batch_idx, batch in enumerate(train_loader):
                        try:
                            # Move batch to device
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)

                            # Zero gradients
                            optimizer.zero_grad()

                            # Forward pass
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )

                            loss = outputs.loss
                            if torch.isnan(loss) or torch.isinf(loss):
                                logger.warning(f"Invalid loss value at batch {batch_idx}, skipping...")
                                continue

                            # Calculate accuracy
                            logits = outputs.logits
                            preds = torch.argmax(logits, dim=1)
                            right_predictions += torch.sum(preds == labels).item()

                            # Backward pass
                            loss.backward()
                            
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            
                            # Optimizer step
                            optimizer.step()
                            scheduler.step()

                            total_train_loss += loss.item()

                            # Log batch progress
                            if batch_idx % 10 == 0:
                                logger.info(
                                    f"{Fore.BLUE}Batch Progress:{Style.RESET_ALL} "
                                    f"{batch_idx}/{len(train_loader)} - "
                                    f"Loss: {loss.item():.4f}"
                                )

                        except Exception as batch_error:
                            logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                            logger.error(f"Batch shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels.shape}")
                            continue  # Skip this batch and continue with next

                    # Validation
                    val_loss, val_accuracy = self._validate_epoch(val_loader)
                    
                    # Report metrics
                    avg_train_loss = total_train_loss / len(train_loader)
                    train_accuracy = right_predictions / len(train_loader.dataset)
                    
                    # Log epoch results
                    logger.info(f"\n{Fore.GREEN}Epoch Results:{Style.RESET_ALL}")
                    logger.info(f"Average training loss: {avg_train_loss:.4f}")
                    logger.info(f"Training accuracy: {train_accuracy:.4f}")
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
                    logger.info(f"Time: {time.time() - start_time:.2f} seconds")

                except Exception as epoch_error:
                    logger.error(f"Error during epoch {epoch + 1}: {str(epoch_error)}")
                    continue  # Skip this epoch and try next

            self.progress["model_trained"] = True
            log_progress("Model Training", ProgressStatus.COMPLETE)
            logger.info("✓ Model training complete")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            log_progress("Model Training", ProgressStatus.FAILED)
            raise

    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation epoch."""
        self.model.eval()
        total_val_loss = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        return avg_val_loss, accuracy

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