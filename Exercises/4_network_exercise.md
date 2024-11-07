# Exercise 4: Network Implementation

## Prerequisites

- Completed Exercise 3
- AIVM devnet running
- Environment activated

## Steps

### **1. Initialize Network**

```python
from food_security_network import FoodSecurityNetwork

# Create network node
network = FoodSecurityNetwork()
print("Network initialized")
```

### **2. Test Collaboration**

```python
import aivm_client as aic

# Prepare and encrypt data
data = "Example demand data"
tokenized_data = aic.tokenize(data)
encrypted_data = aic.BertTinyCryptensor(*tokenized_data)

# Get secure prediction
prediction = await network.predict_demand(encrypted_data)
print("Secure prediction:", prediction)
```

## Success Criteria

- ✓ Network connected
- ✓ Secure predictions working
- ✓ Data sharing enabled
