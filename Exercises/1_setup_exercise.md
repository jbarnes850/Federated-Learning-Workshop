# Exercise 1: Basic Setup

## Objective

Set up your development environment and verify AIVM functionality.

## Prerequisites

- Python 3.8 or higher
- Git installed

## Steps

### **1. Environment Setup**

```bash
# Clone repository
git clone https://github.com/your-org/food-security-workshop
cd food-security-workshop

# Run installation script
chmod +x Examples/1_Basic_Setup/install.sh
./Examples/1_Basic_Setup/install.sh
```

### **2. Start AIVM Devnet**

Open a new terminal and run:

```bash
# Start devnet (keep this terminal open)
aivm-devnet
```

### **3. Verify Installation**

In your original terminal:

```bash
# Run setup validation
python Examples/1_Basic_Setup/test_setup.py
```

### **4. Check Devnet Status**

```bash
# Verify devnet is running
./Examples/utils/manage_devnet.sh status
```

## Success Criteria

- ✓ Environment activated
- ✓ Dependencies installed
- ✓ Devnet running
- ✓ AIVM client connected

## Troubleshooting

If you encounter issues:

1. Ensure devnet is running in a separate terminal
2. Check environment variables in .env
3. Verify Python version compatibility
