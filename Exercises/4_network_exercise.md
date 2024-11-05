# Exercise 4: Network Implementation

## Objective

Set up a basic food security network node.

## Steps

### **Initialize Network**

```python
from food_security_network import FoodSecurityNetwork

# Create network node
network = FoodSecurityNetwork()
print("Network initialized")
```

### **Test Collaboration**

```python
# Test secure prediction
data = "Example demand data"
prediction = await network.predict_demand(data)
print("Secure prediction:", prediction)
```

## Success Criteria

- ✓ Network connected
- ✓ Secure predictions working
- ✓ Data sharing enabled
