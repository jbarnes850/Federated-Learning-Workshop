# Exercise 3: Model Development

## Prerequisites

- Completed Exercise 2
- AIVM devnet running
- Environment activated

## Steps

### **1. Initialize Model**

```python
import aivm_client as aic

# Create and deploy model
model_path = "path/to/model.pth"
aic.upload_bert_tiny_model(model_path, "FoodSecurityBERT")
print("Model deployed")
```

### **2. Test Prediction**

```python
# Test prediction
text = "Need food assistance due to financial difficulties"
tokenized_text = aic.tokenize(text)
encrypted_text = aic.BertTinyCryptensor(*tokenized_text)
prediction = aic.get_prediction(encrypted_text, "FoodSecurityBERT")
print("Prediction:", prediction)
```

## Success Criteria

- ✓ Model deployed
- ✓ Predictions working
- ✓ Privacy maintained
