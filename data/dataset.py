"""
PyTorch Dataset for Election Prediction
Combines sentiment, historical, and demographic features
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ElectionDataset(Dataset):
    """
    PyTorch Dataset combining all features for election prediction.
    
    Features:
    - Sentiment features: (batch, 18) - social media sentiment analysis
    - Historical features: (batch, 24) - past election data
    - Demographic features: (batch, 12) - population characteristics
    
    Labels:
    - Party index (0: LDF, 1: UDF, 2: NDA, 3: OTHERS)
    """
    
    def __init__(
        self,
        sentiment_features: np.ndarray,
        historical_features: np.ndarray,
        demographic_features: np.ndarray,
        labels: np.ndarray,
        booth_ids: Optional[List[str]] = None,
        ward_ids: Optional[List[str]] = None,
        districts: Optional[List[str]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            sentiment_features: Array of shape (N, 18)
            historical_features: Array of shape (N, 24)
            demographic_features: Array of shape (N, 12)
            labels: Array of shape (N,) with party indices
            booth_ids: Optional list of booth identifiers
            ward_ids: Optional list of ward identifiers
            districts: Optional list of district names
        """
        assert len(sentiment_features) == len(historical_features) == \
               len(demographic_features) == len(labels), \
               "All feature arrays must have same length"
        
        self.sentiment_features = torch.FloatTensor(sentiment_features)
        self.historical_features = torch.FloatTensor(historical_features)
        self.demographic_features = torch.FloatTensor(demographic_features)
        self.labels = torch.LongTensor(labels)
        
        self.booth_ids = booth_ids or [f"booth_{i}" for i in range(len(labels))]
        self.ward_ids = ward_ids or [f"ward_{i//5}" for i in range(len(labels))]
        self.districts = districts
        
        self.num_samples = len(labels)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'sentiment': self.sentiment_features[idx],
            'historical': self.historical_features[idx],
            'demographic': self.demographic_features[idx],
            'label': self.labels[idx],
            'booth_id': self.booth_ids[idx],
            'ward_id': self.ward_ids[idx]
        }
    
    @property
    def feature_dims(self) -> Dict[str, int]:
        """Get dimensions of each feature type"""
        return {
            'sentiment': self.sentiment_features.shape[1],
            'historical': self.historical_features.shape[1],
            'demographic': self.demographic_features.shape[1]
        }
    
    @property
    def num_classes(self) -> int:
        """Number of output classes (parties)"""
        return len(torch.unique(self.labels))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader"""
    return {
        'sentiment': torch.stack([item['sentiment'] for item in batch]),
        'historical': torch.stack([item['historical'] for item in batch]),
        'demographic': torch.stack([item['demographic'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'booth_id': [item['booth_id'] for item in batch],
        'ward_id': [item['ward_id'] for item in batch]
    }


def create_mock_dataset(config, correlated: bool = True) -> ElectionDataset:
    """
    Create a mock dataset for demonstration/testing.
    
    Args:
        config: Configuration object
        correlated: If True, features are correlated with labels for realistic training
    
    Returns:
        ElectionDataset with mock data
    """
    from .sentiment_extractor import MockSentimentGenerator
    from .historical_loader import MockHistoricalGenerator
    from .feature_extractor import MockFeatureGenerator
    
    num_booths = config.mock_num_booths
    
    # Generate sentiment features first
    sentiment_gen = MockSentimentGenerator(config)
    sentiment_features = sentiment_gen.generate_booth_sentiments(num_booths)
    
    # Generate labels based on sentiment
    labels = sentiment_gen.generate_labels(num_booths, sentiment_features)
    
    if correlated:
        # Generate correlated features
        historical_gen = MockHistoricalGenerator(config)
        historical_features = historical_gen.generate_with_correlation_to_labels(
            num_booths, labels
        )
        
        feature_gen = MockFeatureGenerator(config)
        demographic_features = feature_gen.generate_with_political_correlation(
            num_booths, labels
        )
    else:
        # Generate independent features
        historical_gen = MockHistoricalGenerator(config)
        historical_features = historical_gen.generate_booth_history(num_booths)
        
        feature_gen = MockFeatureGenerator(config)
        demographic_features = feature_gen.generate_booth_features(num_booths)
    
    # Generate booth/ward IDs and districts
    booths_per_district = num_booths // len(config.districts)
    
    booth_ids = [f"booth_{i:04d}" for i in range(num_booths)]
    ward_ids = [f"ward_{i//5:03d}" for i in range(num_booths)]
    districts = [
        config.districts[min(i // booths_per_district, len(config.districts) - 1)]
        for i in range(num_booths)
    ]
    
    return ElectionDataset(
        sentiment_features=sentiment_features,
        historical_features=historical_features,
        demographic_features=demographic_features,
        labels=labels,
        booth_ids=booth_ids,
        ward_ids=ward_ids,
        districts=districts
    )


def create_data_loaders(
    dataset: ElectionDataset,
    config,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset and create DataLoaders for training, validation, and testing.
    
    Args:
        dataset: Complete ElectionDataset
        config: Configuration object
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# For standalone testing
if __name__ == "__main__":
    from config import Config
    
    config = Config()
    
    print("Creating mock dataset...")
    dataset = create_mock_dataset(config)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimensions: {dataset.feature_dims}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Sentiment shape: {sample['sentiment'].shape}")
    print(f"Historical shape: {sample['historical'].shape}")
    print(f"Demographic shape: {sample['demographic'].shape}")
    print(f"Label: {sample['label']}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(dataset, config)
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
