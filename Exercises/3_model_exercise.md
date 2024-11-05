# Exercise 3: Model Development

## Objective

Deploy a basic BERT model for food security analysis.

## Steps

### **Initialize Model**

```python
from bert_food_security import FoodSecurityBertTiny

# Create model instance
model = FoodSecurityBertTiny(num_labels=6)
print("Model initialized")
```

### **Test Prediction**

```python
# Test prediction
text = "Need food assistance due to financial difficulties"
prediction = model.predict(text)
print("Prediction:", prediction)
```

## Success Criteria

- ✓ Model initialized
- ✓ Predictions working
- ✓ Privacy maintained
