# Exercise 1: Basic Setup

## Objective

Set up your development environment and verify Nillion's AIVM functionality.

## Steps

### **Environment Setup**

```bash
# Clone repository
git clone https://github.com/your-org/food-security-workshop
cd food-security-workshop

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Verify Installation**

```bash
python examples/1_basic_setup/test_setup.py
```

### **Check AIVM Connection**

```python
import aivm_client as aic

# Verify AIVM client
client = aic.AIVMClient()
models = client.get_supported_models()
print("Available models:", models)
```

## Success Criteria

- ✓ Environment activated
- ✓ Dependencies installed
- ✓ AIVM client connected
