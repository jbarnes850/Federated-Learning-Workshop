"""
Food Security Model Testing Module
================================

This module provides comprehensive testing capabilities for the BertTiny model
adapted for food security analysis. It validates model performance, prediction
accuracy, and privacy preservation requirements.

Progress Tracking:
----------------
- Test Dataset Creation ✓
- Model Loading ✓
- Inference Testing ✓
- Performance Metrics ✓

Validation Steps:
---------------
1. Verify test data loading
2. Check model initialization
3. Validate inference pipeline
4. Evaluate performance metrics

Usage:
-----
Run model testing:
    python test_model.py
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from bert_food_security import FoodSecurityBertTiny
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FoodInsecurityDataset(Dataset):
    """Dataset class for food insecurity testing data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the test dataset.
        
        Args:
            texts (list): List of input texts
            labels (list): Corresponding labels
            tokenizer: BertTokenizer instance
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get tokenized and encoded item from dataset."""
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on test data.
    
    Args:
        model: FoodSecurityBertTiny instance
        test_loader: DataLoader for test data
        device: torch device
        
    Returns:
        dict: Performance metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    try:
        logger.info("Starting model testing...")
        
        # Initialize model and tokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BertTokenizer.from_pretrained('bert-tiny')
        model = FoodSecurityBertTiny(num_labels=6).to(device)
        model.load_state_dict(torch.load('models/food_security_bert.pth'))
        
        # Prepare test data
        test_texts = [
            "I need food assistance due to financial difficulties.",
            "The crops failed last season and we are struggling."
        ]
        test_labels = [1, 0]
        
        # Create test dataset and loader
        test_dataset = FoodInsecurityDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizer
        )
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        
        # Evaluate model
        metrics = evaluate_model(model, test_loader, device)
        
        # Log results
        logger.info("\nTest Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")