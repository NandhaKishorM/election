# Kerala Local Body Election Prediction System - Configuration

import os
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    """Configuration for the election prediction model"""
    
    # Paths
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files"))
    
    # Kerala Political Parties
    parties: List[str] = field(default_factory=lambda: [
        "LDF",      # Left Democratic Front (CPI(M) led)
        "UDF",      # United Democratic Front (Congress led)
        "NDA",      # National Democratic Alliance (BJP led)
        "OTHERS"    # Independents and others
    ])
    num_classes: int = 4
    
    # Kerala Districts
    districts: List[str] = field(default_factory=lambda: [
        "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha",
        "Kottayam", "Idukki", "Ernakulam", "Thrissur", "Palakkad",
        "Malappuram", "Kozhikode", "Wayanad", "Kannur", "Kasaragod"
    ])
    
    # Model Architecture
    sentiment_hidden_dim: int = 256
    historical_hidden_dim: int = 128
    demographic_hidden_dim: int = 64
    fusion_hidden_dim: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.3
    
    # Sentiment Encoder (using pretrained model)
    sentiment_model_name: str = "distilbert-base-uncased"
    sentiment_embedding_dim: int = 768
    max_sentiment_length: int = 128
    
    # Historical Encoder
    num_past_elections: int = 3  # 2010, 2015, 2020
    historical_features_per_election: int = 8  # vote_share, turnout, margin, etc.
    historical_input_dim: int = 24  # num_past_elections * historical_features
    
    # Demographic Features
    num_demographic_features: int = 12
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    early_stopping_patience: int = 10
    warmup_steps: int = 100
    
    # Data Split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Device
    device: str = "cuda"  # Will be updated based on availability
    
    # Mock Data Settings (for demonstration)
    mock_num_booths: int = 1000
    mock_num_wards: int = 200
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)


# Sentiment keywords for social media extraction
SENTIMENT_KEYWORDS = {
    "LDF": [
        "LDF", "CPM", "CPI", "Pinarayi", "Left Front", "Communist",
        "LDF Kerala", "ഇടത്", "പിണറായി", "സിപിഎം"
    ],
    "UDF": [
        "UDF", "Congress", "IUML", "Oommen Chandy", "VD Satheesan",
        "United Democratic Front", "കോൺഗ്രസ്", "യുഡിഎഫ്"
    ],
    "NDA": [
        "NDA", "BJP", "BJP Kerala", "Surendran", "Kummanam",
        "National Democratic Alliance", "ബിജെപി", "എൻഡിഎ"
    ],
    "general": [
        "Kerala election", "local body election", "panchayat election",
        "municipality election", "corporation election",
        "കേരള തെരഞ്ഞെടുപ്പ്", "തദ്ദേശ തെരഞ്ഞെടുപ്പ്"
    ]
}

# Historical election features
HISTORICAL_FEATURES = [
    "vote_share",           # Percentage of votes received
    "voter_turnout",        # Percentage of eligible voters who voted
    "victory_margin",       # Margin of victory (positive) or defeat (negative)
    "num_candidates",       # Number of candidates in the booth
    "incumbent_vote_change", # Change in incumbent party's vote share
    "swing",                # Vote swing from previous election
    "valid_votes",          # Number of valid votes
    "rejected_votes"        # Number of rejected votes
]

# Demographic features
DEMOGRAPHIC_FEATURES = [
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
    "senior_citizen_percentage"
]
