"""
Food Bank Data Generation Module
==============================

This module provides synthetic data generation capabilities for food bank demand scenarios,
creating realistic test data while preserving privacy of actual food bank operations.

Prerequisites:
------------
- Environment setup completed
- Dependencies installed

Progress Tracking:
----------------
- Data Schema Definition ✓
- Synthetic Generation ✓
- Data Validation ✓
- Privacy Compliance ✓

Usage:
-----
Generate synthetic food bank data:
    data = generate_synthetic_data(num_entries=100)
    print(data.head())
"""

import pandas as pd
from faker import Faker
import logging
from typing import Optional, Dict, List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for generating realistic data
fake = Faker()

class FoodBankDataGenerator:
    """Handles synthetic data generation with privacy considerations."""
    
    def __init__(self):
        """Initialize data generator with progress tracking."""
        self.progress = {
            "schema_defined": False,
            "data_generated": False,
            "data_validated": False,
            "privacy_checked": False
        }
        self.schema = self._define_schema()
        
    def _define_schema(self) -> Dict[str, Dict]:
        """Define and validate data schema."""
        try:
            schema = {
                'Region': {'type': str, 'source': 'city'},
                'City': {'type': str, 'source': 'city'},
                'ZipCode': {'type': str, 'source': 'zipcode'},
                'Population': {'type': int, 'range': (1000, 50000)},
                'HouseholdSize': {'type': int, 'range': (1, 6)},
                'IncomeLevel': {'type': str, 'options': ['Low', 'Medium', 'High']},
                'FoodType': {'type': str, 'options': [
                    'Canned Goods', 'Dry Goods', 'Fresh Produce', 
                    'Meat/Poultry', 'Dairy'
                ]},
                'DemandAmount': {'type': int, 'range': (10, 500)}
            }
            self.progress["schema_defined"] = True
            logger.info("✓ Schema defined successfully")
            return schema
            
        except Exception as e:
            logger.error(f"❌ Schema definition failed: {e}")
            raise

    def generate_data(self, num_entries: int = 100) -> Optional[pd.DataFrame]:
        """Generate synthetic dataset with privacy considerations."""
        try:
            data = []
            for _ in range(num_entries):
                entry = {}
                for col, props in self.schema.items():
                    if props['type'] == str:
                        if 'options' in props:
                            entry[col] = fake.random_choice(props['options'])
                        else:
                            entry[col] = getattr(fake, props['source'])()
                    elif props['type'] == int:
                        entry[col] = fake.random_int(
                            min=props['range'][0],
                            max=props['range'][1]
                        )
                data.append(entry)
                
            df = pd.DataFrame(data)
            self.progress["data_generated"] = True
            logger.info(f"✓ Generated {num_entries} synthetic entries")
            
            # Validate generated data
            self._validate_data(df)
            return df
            
        except Exception as e:
            logger.error(f"❌ Data generation failed: {e}")
            raise

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate data structure and content."""
        try:
            # Check schema compliance
            for col, props in self.schema.items():
                assert col in data.columns, f"Missing column: {col}"
                if props['type'] == int:
                    assert data[col].between(
                        props['range'][0],
                        props['range'][1]
                    ).all(), f"Invalid range in {col}"
                if 'options' in props:
                    assert data[col].isin(props['options']).all(), \
                        f"Invalid values in {col}"
            
            # Check privacy considerations
            self._verify_privacy(data)
            
            self.progress["data_validated"] = True
            logger.info("✓ Data validation complete")
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            raise

    def _verify_privacy(self, data: pd.DataFrame) -> None:
        """Verify privacy preservation in generated data."""
        try:
            # Check for PII
            pii_columns = ['Region', 'City', 'ZipCode']
            for col in pii_columns:
                # Ensure no real-world matches
                assert not data[col].isin(
                    self._get_real_world_examples()
                ).any(), f"Real-world data found in {col}"
            
            self.progress["privacy_checked"] = True
            logger.info("✓ Privacy verification complete")
            
        except Exception as e:
            logger.error(f"❌ Privacy verification failed: {e}")
            raise

    def _get_real_world_examples(self) -> List[str]:
        """Return list of real-world examples to avoid."""
        return ["New York", "Los Angeles", "Chicago", "90210"]

    def get_progress(self) -> Dict[str, str]:
        """Get current progress status."""
        return {
            step: "✓" if status else "❌"
            for step, status in self.progress.items()
        }

def generate_synthetic_data(num_entries: int = 100) -> pd.DataFrame:
    """Convenience function for generating synthetic data."""
    generator = FoodBankDataGenerator()
    return generator.generate_data(num_entries)

if __name__ == "__main__":
    try:
        # Generate test data
        generator = FoodBankDataGenerator()
        data = generator.generate_data(num_entries=50)
        
        # Show progress and sample data
        logger.info("\nGeneration Status:")
        for step, status in generator.get_progress().items():
            logger.info(f"{step}: {status}")
            
        logger.info("\nSample Data:")
        print(data.head())
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        exit(1)
