# Exercise 2: Data Privacy

## Objective

Implement basic data encryption and privacy preservation using Nillion's AIVM.

## Steps

### **Generate Test Data**

```python
from food_bank_data import generate_synthetic_data

# Generate sample data
data = generate_synthetic_data(num_entries=10)
print("Sample data:", data.head())
```

### **Encrypt Data**

```python
import aivm_client as aic

# Encrypt sensitive data
encrypted_data = aic.BertTinyCryptensor(data.to_json())
print("Data encrypted successfully")
```

## Success Criteria

- ✓ Data generated
- ✓ Encryption working
- ✓ Privacy preserved
