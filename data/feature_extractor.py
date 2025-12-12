"""
Demographic and Geo-Political Feature Extractor
Extracts additional features for election prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DemographicData:
    """Demographic data for a booth/ward"""
    booth_id: str
    ward_id: str
    district: str
    
    # Population metrics
    population: int
    population_density: float
    
    # Education
    literacy_rate: float
    
    # Urban/Rural
    is_urban: bool
    urban_rural_ratio: float
    
    # Demographics
    male_female_ratio: float
    sc_st_percentage: float
    minority_percentage: float
    
    # Economic
    avg_income_level: float  # Normalized 0-1
    unemployment_rate: float
    agriculture_percentage: float
    service_sector_percentage: float
    
    # Age distribution
    youth_percentage: float  # 18-35
    senior_percentage: float  # 60+


class FeatureExtractor:
    """
    Extract demographic and geo-political features.
    In production, would load from census data and other sources.
    """
    
    def __init__(self, config, data_path: Optional[str] = None):
        self.config = config
        self.data_path = data_path
    
    def load_demographic_data(self) -> pd.DataFrame:
        """Load demographic data from file"""
        import os
        
        if self.data_path:
            file_path = os.path.join(self.data_path, "demographics.csv")
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
        
        return None
    
    def extract_features(self, booth_id: str) -> np.ndarray:
        """Extract all demographic features for a booth"""
        df = self.load_demographic_data()
        
        if df is None or booth_id not in df['booth_id'].values:
            return np.zeros(self.config.num_demographic_features)
        
        row = df[df['booth_id'] == booth_id].iloc[0]
        
        features = [
            row.get('population_density', 0),
            row.get('literacy_rate', 0),
            row.get('urban_rural_ratio', 0),
            row.get('male_female_ratio', 0),
            row.get('sc_st_percentage', 0),
            row.get('minority_percentage', 0),
            row.get('avg_income_level', 0),
            row.get('unemployment_rate', 0),
            row.get('agriculture_percentage', 0),
            row.get('service_sector_percentage', 0),
            row.get('youth_percentage', 0),
            row.get('senior_percentage', 0)
        ]
        
        return np.array(features, dtype=np.float32)


class MockFeatureGenerator:
    """
    Generate mock demographic features for demonstration.
    Simulates Kerala's demographic patterns.
    """
    
    def __init__(self, config):
        self.config = config
        self.districts = config.districts
        
        # District characteristics (rough approximations)
        self.district_profiles = {
            "Thiruvananthapuram": {"urban": 0.7, "literacy": 0.93, "minority": 0.25},
            "Kollam": {"urban": 0.5, "literacy": 0.94, "minority": 0.20},
            "Pathanamthitta": {"urban": 0.3, "literacy": 0.96, "minority": 0.45},
            "Alappuzha": {"urban": 0.5, "literacy": 0.96, "minority": 0.22},
            "Kottayam": {"urban": 0.4, "literacy": 0.97, "minority": 0.50},
            "Idukki": {"urban": 0.2, "literacy": 0.92, "minority": 0.40},
            "Ernakulam": {"urban": 0.8, "literacy": 0.95, "minority": 0.30},
            "Thrissur": {"urban": 0.5, "literacy": 0.95, "minority": 0.25},
            "Palakkad": {"urban": 0.4, "literacy": 0.89, "minority": 0.35},
            "Malappuram": {"urban": 0.4, "literacy": 0.93, "minority": 0.70},
            "Kozhikode": {"urban": 0.6, "literacy": 0.96, "minority": 0.40},
            "Wayanad": {"urban": 0.2, "literacy": 0.89, "minority": 0.30},
            "Kannur": {"urban": 0.5, "literacy": 0.95, "minority": 0.30},
            "Kasaragod": {"urban": 0.3, "literacy": 0.90, "minority": 0.55}
        }
    
    def generate_booth_features(
        self,
        num_booths: int = 1000
    ) -> np.ndarray:
        """
        Generate mock demographic features for booths.
        
        Features (12 total):
        - population_density (normalized)
        - literacy_rate
        - urban_rural_ratio
        - male_female_ratio
        - sc_st_percentage
        - minority_percentage
        - avg_income_level
        - unemployment_rate
        - agriculture_percentage
        - service_sector_percentage
        - youth_percentage
        - senior_percentage
        """
        np.random.seed(45)
        
        num_features = self.config.num_demographic_features  # 12
        features = np.zeros((num_booths, num_features))
        
        # Assign booths to districts roughly equally
        booths_per_district = num_booths // len(self.districts)
        
        for i in range(num_booths):
            district_idx = i // booths_per_district
            if district_idx >= len(self.districts):
                district_idx = len(self.districts) - 1
            district = self.districts[district_idx]
            profile = self.district_profiles.get(
                district, 
                {"urban": 0.5, "literacy": 0.93, "minority": 0.30}
            )
            
            # Population density (normalized 0-1)
            # Urban areas have higher density
            base_density = profile["urban"] * 0.6 + 0.2
            features[i, 0] = np.clip(
                base_density + np.random.normal(0, 0.15), 0.1, 1.0
            )
            
            # Literacy rate (Kerala has high literacy)
            features[i, 1] = np.clip(
                profile["literacy"] + np.random.normal(0, 0.02), 0.8, 1.0
            )
            
            # Urban-rural ratio
            features[i, 2] = np.clip(
                profile["urban"] + np.random.normal(0, 0.1), 0.1, 0.9
            )
            
            # Male-female ratio (Kerala has more females typically)
            features[i, 3] = np.clip(
                np.random.normal(0.96, 0.02), 0.9, 1.05
            )
            
            # SC/ST percentage (varies by district)
            sc_st_base = 0.12 if district in ["Wayanad", "Idukki", "Palakkad"] else 0.08
            features[i, 4] = np.clip(
                sc_st_base + np.random.normal(0, 0.03), 0.02, 0.25
            )
            
            # Minority percentage
            features[i, 5] = np.clip(
                profile["minority"] + np.random.normal(0, 0.05), 0.1, 0.8
            )
            
            # Average income level (normalized)
            income_base = 0.6 if profile["urban"] > 0.5 else 0.4
            features[i, 6] = np.clip(
                income_base + np.random.normal(0, 0.1), 0.2, 0.9
            )
            
            # Unemployment rate (Kerala has educated unemployment issue)
            features[i, 7] = np.clip(
                np.random.beta(2, 8) * 0.3 + 0.05, 0.03, 0.25
            )
            
            # Agriculture percentage (inverse of urban)
            features[i, 8] = np.clip(
                (1 - profile["urban"]) * 0.6 + np.random.normal(0, 0.1), 0.1, 0.7
            )
            
            # Service sector percentage
            features[i, 9] = np.clip(
                profile["urban"] * 0.5 + np.random.normal(0.2, 0.1), 0.2, 0.7
            )
            
            # Youth percentage (18-35)
            features[i, 10] = np.clip(
                np.random.normal(0.30, 0.05), 0.2, 0.4
            )
            
            # Senior citizen percentage (60+)
            features[i, 11] = np.clip(
                np.random.normal(0.15, 0.03), 0.08, 0.25
            )
        
        return features.astype(np.float32)
    
    def generate_with_political_correlation(
        self,
        num_booths: int = 1000,
        labels: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate features with some correlation to political outcomes.
        
        Kerala political patterns (simplified):
        - LDF: Stronger in working-class, SC/ST areas
        - UDF: Stronger in minority-heavy, Christian areas
        - NDA: Stronger in urban Hindu-majority areas
        """
        features = self.generate_booth_features(num_booths)
        
        if labels is not None:
            for i in range(num_booths):
                winner = labels[i]
                
                if winner == 0:  # LDF
                    # Slightly lower income, higher agriculture
                    features[i, 6] *= 0.9  # income
                    features[i, 8] *= 1.1  # agriculture
                    features[i, 4] *= 1.2  # SC/ST
                    
                elif winner == 1:  # UDF
                    # Higher minority percentage
                    features[i, 5] *= 1.15  # minority
                    features[i, 1] *= 1.02  # literacy
                    
                elif winner == 2:  # NDA
                    # More urban, higher income
                    features[i, 2] *= 1.1  # urban
                    features[i, 6] *= 1.1  # income
                    features[i, 5] *= 0.85  # minority (lower)
                
                # Clip values to valid ranges
                features[i] = np.clip(features[i], 0, 1)
        
        return features


def get_demographic_feature_names() -> List[str]:
    """Get names of demographic features"""
    return [
        "population_density",
        "literacy_rate",
        "urban_rural_ratio",
        "male_female_ratio",
        "sc_st_percentage",
        "minority_percentage",
        "avg_income_level",
        "unemployment_rate",
        "agriculture_percentage",
        "service_sector_percentage",
        "youth_percentage",
        "senior_percentage"
    ]
