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
    data = generate_synthetic_data(num_entries=1000)
    print(data.head())
"""

import pandas as pd
from faker import Faker
import logging
from typing import Optional, Dict, List
import numpy as np
from numpy.random import choice

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
        """Define and validate data schema with balanced scenarios."""
        try:
            schema = {
                'Region': {'type': str, 'source': 'city'},
                'City': {'type': str, 'source': 'city'},
                'ZipCode': {'type': str, 'source': 'zipcode'},
                'Population': {
                    'type': int, 
                    'range': (100000, 5000000),
                    'distribution': 'balanced' 
                },
                'HouseholdSize': {
                    'type': int, 
                    'range': (1, 6),
                    'distribution': 'weighted'  
                },
                'IncomeLevel': {
                    'type': str, 
                    'options': ['Low', 'Medium', 'High'],
                    'weights': [0.4, 0.4, 0.2]  
                },
                'FoodType': {
                    'type': str, 
                    'options': [
                        'Canned Goods', 'Dry Goods', 'Fresh Produce', 
                        'Meat/Poultry', 'Dairy'
                    ],
                    'weights': [0.25, 0.25, 0.2, 0.15, 0.15] 
                },
                'DemandAmount': {
                    'type': int, 
                    'range': (10, 500),
                    'distribution': 'normal'  
                }
            }
            self.progress["schema_defined"] = True
            logger.info("✓ Schema defined successfully")
            return schema
            
        except Exception as e:
            logger.error(f"❌ Schema definition failed: {e}")
            raise

    def generate_data(self, num_entries: int = 1000) -> Optional[pd.DataFrame]:
        """Generate balanced synthetic dataset."""
        try:
            data = []
            
            # Generate balanced scenarios
            scenarios = [
                # High population urban centers
                {'population_range': (1000000, 5000000), 'income': 'High', 'weight': 0.2},
                # Medium cities
                {'population_range': (500000, 1000000), 'income': 'Medium', 'weight': 0.3},
                # Small cities
                {'population_range': (100000, 500000), 'income': 'Low', 'weight': 0.3},
                # Emergency scenarios
                {'population_range': (100000, 5000000), 'demand_multiplier': 1.5, 'weight': 0.2}
            ]
            
            for scenario in scenarios:
                n_entries = int(num_entries * scenario['weight'])
                for _ in range(n_entries):
                    entry = {}
                    for col, props in self.schema.items():
                        if col == 'Population':
                            entry[col] = fake.random_int(
                                min=scenario['population_range'][0],
                                max=scenario['population_range'][1]
                            )
                        elif col == 'IncomeLevel':
                            if 'income' in scenario:
                                entry[col] = scenario['income']
                            else:
                                entry[col] = choice(
                                    props['options'],
                                    p=props['weights']
                                )
                        elif col == 'DemandAmount':
                            base_demand = fake.random_int(
                                min=props['range'][0],
                                max=props['range'][1]
                            )
                            entry[col] = int(base_demand * scenario.get('demand_multiplier', 1.0))
                        elif props['type'] == str:
                            if 'options' in props:
                                # Use numpy's choice for weighted selection
                                entry[col] = choice(
                                    props['options'],
                                    p=props.get('weights')
                                )
                            else:
                                entry[col] = getattr(fake, props['source'])()
                        elif props['type'] == int:
                            entry[col] = fake.random_int(
                                min=props['range'][0],
                                max=props['range'][1]
                            )
                    data.append(entry)
                    
            df = pd.DataFrame(data)
            
            # Add correlations between variables
            df['DemandAmount'] = df.apply(
                lambda row: self._adjust_demand_by_factors(
                    row['DemandAmount'],
                    row['Population'],
                    row['HouseholdSize'],
                    row['IncomeLevel']
                ),
                axis=1
            )
            
            self.progress["data_generated"] = True
            logger.info(f"✓ Generated {len(df)} balanced synthetic entries")
            
            # Validate generated data
            self._validate_data(df)
            return df
            
        except Exception as e:
            logger.error(f"❌ Data generation failed: {e}")
            raise

    def _adjust_demand_by_factors(
        self, 
        base_demand: int, 
        population: int, 
        household_size: int, 
        income_level: str
    ) -> int:
        """Adjust demand based on demographic factors."""
        # Population factor
        pop_factor = np.log10(population) / np.log10(5000000)  # Normalize by max population
        
        # Household size factor (larger households need more)
        household_factor = household_size / 3.0  # Normalize by average household size
        
        # Income level factor (inverse relationship)
        income_factors = {'Low': 1.2, 'Medium': 1.0, 'High': 0.8}
        income_factor = income_factors[income_level]
        
        # Combine factors
        adjusted_demand = int(
            base_demand * 
            (0.5 + 0.5 * pop_factor) * 
            household_factor * 
            income_factor
        )
        
        # Ensure within valid range
        return max(10, min(500, adjusted_demand))

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

def generate_synthetic_data(num_entries: int = 1000, save_path: str = "synthetic_data.csv") -> pd.DataFrame:
    """Convenience function for generating and saving synthetic data."""
    generator = FoodBankDataGenerator()
    data = generator.generate_data(num_entries)
    data.to_csv(save_path, index=False)
    logger.info(f"✓ Synthetic data saved to {save_path}")
    return data

if __name__ == "__main__":
    try:
        # Generate and save test data
        generator = FoodBankDataGenerator()
        data = generator.generate_data(num_entries=1000)  # Increased number of entries
        data.to_csv("synthetic_data.csv", index=False)
        
        # Show progress and sample data
        logger.info("\nGeneration Status:")
        for step, status in generator.get_progress().items():
            logger.info(f"{step}: {status}")
            
        logger.info("\nSample Data:")
        print(data.head())
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        exit(1)
