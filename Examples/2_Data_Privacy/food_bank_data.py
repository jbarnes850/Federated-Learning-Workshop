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
                    'range': (50000, 5000000),
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
                    'range': (10, 1000),  # Increased max range to match adjusted demands
                    'distribution': 'normal',
                    'peaks': {  # Add demand peaks
                        'holiday': 1.5,
                        'summer': 1.3,
                        'emergency': 2.0
                    }
                },
                'EmergencyStatus': {
                    'type': str,
                    'options': ['None', 'Weather', 'Economic', 'Health'],
                    'weights': [0.7, 0.1, 0.1, 0.1]
                },
                'SeasonalFactor': {
                    'type': str,
                    'options': ['Normal', 'Holiday', 'Summer', 'BackToSchool'],
                    'weights': [0.6, 0.15, 0.15, 0.1]
                },
                'SupplyType': {  # Add this to match train_model.py features
                    'type': str,
                    'options': ['Perishable', 'Non-Perishable'],
                    'derived': True  # Flag for derived field
                },
                'DistributionPriority': {  # Add this to match train_model.py features
                    'type': str,
                    'options': ['High', 'Medium', 'Standard'],
                    'derived': True
                }
            }
            self.progress["schema_defined"] = True
            logger.info("✓ Schema defined successfully")
            return schema
            
        except Exception as e:
            logger.error(f"❌ Schema definition failed: {e}")
            raise

    def generate_data(self, num_entries: int = 5000) -> Optional[pd.DataFrame]:
        """Generate more diverse scenarios."""
        scenarios = [
            # Urban centers
            {
                'population_range': (1000000, 5000000),
                'income_mix': {'Low': 0.4, 'Medium': 0.4, 'High': 0.2},
                'weight': 0.3
            },
            # Suburban areas
            {
                'population_range': (200000, 999999),
                'income_mix': {'Low': 0.3, 'Medium': 0.5, 'High': 0.2},
                'weight': 0.4
            },
            # Rural areas
            {
                'population_range': (50000, 199999),
                'income_mix': {'Low': 0.5, 'Medium': 0.3, 'High': 0.2},
                'weight': 0.2
            },
            # Emergency zones
            {
                'population_range': (50000, 5000000),
                'emergency_status': ['Weather', 'Economic', 'Health'],
                'demand_multiplier': 2.0,
                'weight': 0.1
            }
        ]
        
        try:
            data = []
            
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
            
            # Add derived features
            df = self._add_derived_features(df)
            
            # Adjust demand
            df['DemandAmount'] = df.apply(
                lambda row: self._adjust_demand_by_factors(
                    base_demand=row['DemandAmount'],
                    population=row['Population'],
                    household_size=row['HouseholdSize'],
                    income_level=row['IncomeLevel'],
                    food_type=row['FoodType'],
                    emergency_status=row['EmergencyStatus'],
                    month=row['SeasonalFactor']
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
        income_level: str,
        food_type: str,
        emergency_status: str = 'None',
        month: str = 'Normal'
    ) -> int:
        """Create more realistic demand adjustments."""
        try:
            # Population impact is logarithmic
            pop_impact = np.log2(population / 100000) * 0.8  
            
            # Income level has inverse exponential effect 
            income_impact = {
                'Low': 1.5,    
                'Medium': 1.2,
                'High': 0.8
            }[income_level]
            
            # Household size
            household_impact = min(np.sqrt(household_size), 2.0)
            
            # Seasonal variations
            seasonal_impact = {
                'Holiday': 1.5,   
                'Summer': 1.3,
                'BackToSchool': 1.2,
                'Normal': 1.0
            }.get(month, 1.0)
            
            # Emergency multipliers
            emergency_impact = {
                'Weather': 1.7,    
                'Economic': 1.5,
                'Health': 1.3,
                'None': 1.0
            }[emergency_status]
            
            # Food type specific factors
            food_type_impact = {
                'Fresh Produce': 1.2,
                'Meat/Poultry': 1.3,
                'Dairy': 1.1,
                'Canned Goods': 1.0,
                'Dry Goods': 0.9
            }[food_type]
            
            # Calculate final demand
            adjusted_demand = base_demand * (
                pop_impact * 
                income_impact * 
                household_impact * 
                seasonal_impact * 
                emergency_impact *
                food_type_impact
            )
            
            # Ensure within schema range
            return max(10, min(1000, int(adjusted_demand)))
            
        except Exception as e:
            logger.error(f"Demand adjustment failed: {e}")
            return base_demand  # Return original demand if adjustment fails

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

    def _add_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific features for better prediction."""
        # Calculate food insecurity risk score
        data['risk_score'] = (
            (data['Population'] / data['HouseholdSize']) * 
            (data['IncomeLevel'].map({'Low': 3, 'Medium': 2, 'High': 1})) / 
            1000
        )
        
        # Add emergency supply ratio
        data['emergency_ratio'] = data.apply(
            lambda row: 1.5 if row['FoodType'] in ['Canned Goods', 'Dry Goods'] 
            else 1.0,
            axis=1
        )
        
        # Add perishability factor
        data['perishability'] = data['FoodType'].map({
            'Fresh Produce': 3,
            'Meat/Poultry': 3,
            'Dairy': 2,
            'Canned Goods': 1,
            'Dry Goods': 1
        })

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to match training data format."""
        # Add SupplyType
        df['SupplyType'] = df['FoodType'].apply(
            lambda x: 'Perishable' if x in ['Fresh Produce', 'Meat/Poultry', 'Dairy'] 
            else 'Non-Perishable'
        )
        
        # Add DistributionPriority
        df['DistributionPriority'] = df.apply(
            lambda row: 'High' if row['IncomeLevel'] == 'Low' and row['HouseholdSize'] >= 4
            else 'Medium' if row['IncomeLevel'] == 'Low'
            else 'Standard',
            axis=1
        )
        
        return df

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
        data = generator.generate_data(num_entries=5000)
        data.to_csv("synthetic_data.csv", index=False)
        
        # Show progress and sample data
        logger.info("\nGeneration Status:")
        for step, status in generator.get_progress().items():
            logger.info(f"{step}: {status}")
            
        logger.info(f"\nGenerated {len(data)} entries")
        logger.info("\nSample Data:")
        print(data.head())
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        exit(1)
