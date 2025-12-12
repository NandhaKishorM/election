"""
Kerala Local Body Election Prediction 2025
Simplified Training Script - Loads CSV Data Files

Run create_dataset.py first to generate the CSV files, then run this script.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration"""
    parties: List[str] = field(default_factory=lambda: ["LDF", "UDF", "NDA", "OTHERS"])
    num_classes: int = 4
    hidden_dim: int = 256
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ElectionDataset(Dataset):
    """Load election data from CSV files"""
    
    def __init__(self, data_dir: str = "data_files"):
        self.config = Config()
        self.party_to_idx = {"LDF": 0, "UDF": 1, "NDA": 2, "OTHERS": 3}
        
        # Load CSV files
        print("Loading CSV data files...")
        self.wards_df = pd.read_csv(os.path.join(data_dir, "kerala_election_wards.csv"))
        self.sentiment_df = pd.read_csv(os.path.join(data_dir, "kerala_sentiment_2025.csv"))
        self.demographics_df = pd.read_csv(os.path.join(data_dir, "kerala_demographics.csv"))
        
        print(f"  Loaded {len(self.wards_df)} ward records")
        print(f"  Loaded sentiment for {len(self.sentiment_df)} parties")
        print(f"  Loaded demographics for {len(self.demographics_df)} districts")
        
        # Process into features
        self.features, self.labels, self.meta = self._process_data()
        print(f"  Created {len(self.features)} samples")
    
    def _process_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Process CSV data into model features"""
        
        # Create sentiment lookup
        sentiment_lookup = {}
        for _, row in self.sentiment_df.iterrows():
            sentiment_lookup[row['party']] = row['final_sentiment_score']
        
        features_list = []
        labels_list = []
        meta_list = []
        
        for _, row in self.wards_df.iterrows():
            features = []
            
            # Sentiment features (18) - 4 parties * 4 features + 2 global
            for party in self.config.parties:
                sent = sentiment_lookup.get(party, 0)
                features.extend([
                    sent,                    # sentiment
                    abs(sent) * 0.3,        # std
                    0.5 + sent * 0.3,       # mentions
                    sent * 0.1              # trend
                ])
            features.extend([0.5, 0.2])     # engagement, volatility
            
            # Historical features (24) - 3 elections * 8 features
            for year in ['2015', '2018', '2020']:
                winner = row.get(f'winner_{year}', row['winner_2020'])
                vote_share = row.get(f'vote_share_{year}', row['vote_share_2020'])
                turnout = row.get(f'turnout_{year}', row['turnout_2020'])
                
                # Vote shares per party
                for party in self.config.parties:
                    if party == winner:
                        features.append(vote_share)
                    else:
                        features.append((1 - vote_share) / 3)
                
                features.append(turnout)
                features.append(row.get('margin_2020', 0.1))
                features.append(0.0)  # swing
                features.append(1.0 if winner == row['winner_2015'] else 0.0)
            
            # Demographic features (12)
            features.extend([
                row['population_density'] / 2000,  # normalize
                row['literacy_rate'] / 100,
                row['urban_pct'] / 100,
                1.0,  # gender ratio
                row['sc_st_pct'] / 100,
                (row['muslim_pct'] + row['christian_pct']) / 100,  # minority
                0.8,  # development index
                0.08,  # unemployment
                (100 - row['urban_pct']) / 100,  # agriculture
                row['urban_pct'] / 100,  # service
                0.30,  # youth
                0.12   # senior
            ])
            
            features_list.append(features)
            
            # Label: Predict 2025 winner based on swing model
            label = self._predict_2025(row)
            labels_list.append(self.party_to_idx[label])
            
            meta_list.append({
                'ward_id': row['ward_id'],
                'district': row['district'],
                'body_type': row['body_type']
            })
        
        return np.array(features_list, dtype=np.float32), np.array(labels_list), meta_list
    
    def _predict_2025(self, row) -> str:
        """Predict 2025 winner using swing model"""
        
        winner_2020 = row['winner_2020']
        ls2024_winner = row['ls2024_winner']
        
        # Base probability from 2020 with retention
        base_retention = 0.60
        
        # LS 2024 momentum boost
        ls_boost = 0.15 if ls2024_winner != winner_2020 else 0
        
        probs = {"LDF": 0.0, "UDF": 0.0, "NDA": 0.0, "OTHERS": 0.0}
        
        # Start with 2020 winner advantage
        for party in probs:
            if party == winner_2020:
                probs[party] = base_retention
            else:
                probs[party] = (1 - base_retention) / 3
        
        # Apply LS 2024 momentum
        if ls2024_winner in probs:
            probs[ls2024_winner] += ls_boost
            
        # District-specific adjustments
        district = row['district']
        if district == "Malappuram":
            probs["UDF"] *= 1.2  # Strong IUML base
        elif district == "Kannur":
            probs["LDF"] *= 1.15  # Strong CPM base
        elif district == "Thrissur":
            probs["NDA"] *= 1.4  # First LS seat momentum
        elif district in ["Pathanamthitta", "Kottayam"]:
            probs["UDF"] *= 1.1  # Christian majority
        
        # Normalize
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        return np.random.choice(list(probs.keys()), p=list(probs.values()))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'label': torch.LongTensor([self.labels[idx]])[0],
            'meta': self.meta[idx]
        }


class ElectionModel(nn.Module):
    """Election prediction model"""
    
    def __init__(self, input_dim: int = 54, hidden_dim: int = 256, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        logits = self.classifier(encoded)
        conf = self.confidence(encoded)
        return {'logits': logits, 'probs': torch.softmax(logits, dim=-1), 'confidence': conf}


def train(model, train_loader, val_loader, config):
    """Training loop"""
    
    device = config.device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_acc = 0
    patience = 0
    
    print(f"\n🚀 Training on {device}")
    print("-" * 50)
    
    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch['features'].to(device))
            loss = criterion(output['logits'], batch['label'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (output['logits'].argmax(-1) == batch['label'].to(device)).sum().item()
            train_total += len(batch['label'])
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['features'].to(device))
                loss = criterion(output['logits'], batch['label'].to(device))
                val_loss += loss.item()
                val_correct += (output['logits'].argmax(-1) == batch['label'].to(device)).sum().item()
                val_total += len(batch['label'])
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss / len(val_loader))
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            patience = 0
            mark = "✓"
        else:
            patience += 1
            mark = ""
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} {mark}")
        
        if patience >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\n✓ Best accuracy: {best_acc:.4f}")
    return model


def predict(model, dataset, config):
    """Generate predictions"""
    
    device = config.device
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    results = []
    
    print("\n🗳️ Generating predictions...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            output = model(batch['features'].to(device))
            preds = output['logits'].argmax(-1).cpu().numpy()
            probs = output['probs'].cpu().numpy()
            confs = output['confidence'].cpu().numpy()
            
            for i in range(len(preds)):
                meta = batch['meta']
                results.append({
                    'ward_id': meta['ward_id'][i] if isinstance(meta['ward_id'], list) else meta['ward_id'],
                    'district': meta['district'][i] if isinstance(meta['district'], list) else meta['district'],
                    'body_type': meta['body_type'][i] if isinstance(meta['body_type'], list) else meta['body_type'],
                    'predicted': config.parties[preds[i]],
                    'confidence': float(confs[i]),
                    'LDF': float(probs[i, 0]),
                    'UDF': float(probs[i, 1]),
                    'NDA': float(probs[i, 2]),
                    'OTHERS': float(probs[i, 3])
                })
    
    return pd.DataFrame(results)


def print_summary(results, config):
    """Print election summary"""
    
    print("\n" + "=" * 60)
    print("🗳️  KERALA LOCAL BODY ELECTION 2025 PREDICTION")
    print("=" * 60)
    
    total = len(results)
    counts = results['predicted'].value_counts()
    
    print(f"\n📊 SEAT PROJECTION (Total: {total}):")
    print("-" * 40)
    
    for party in config.parties:
        n = counts.get(party, 0)
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        emoji = {"LDF": "🔴", "UDF": "🔵", "NDA": "🟠", "OTHERS": "⚪"}[party]
        print(f"{emoji} {party:6s}: {n:4d} ({pct:5.1f}%) {bar}")
    
    winner = counts.idxmax()
    print(f"\n🏆 PROJECTED WINNER: {winner}")
    
    print("\n📍 DISTRICT-WISE:")
    print("-" * 40)
    district_counts = results.groupby(['district', 'predicted']).size().unstack(fill_value=0)
    district_counts['Winner'] = district_counts.idxmax(axis=1)
    print(district_counts)
    
    print("\n" + "=" * 60)


def main():
    """Main"""
    
    print("\n" + "=" * 60)
    print("🗳️  KERALA ELECTION 2025 PREDICTION")
    print("=" * 60)
    
    config = Config()
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check if data files exist
    if not os.path.exists("data_files/kerala_election_wards.csv"):
        print("\n⚠️  Data files not found!")
        print("   Run: python create_dataset.py")
        return
    
    # Load data
    dataset = ElectionDataset("data_files")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], 
                                     generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    # Create model
    model = ElectionModel()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model = train(model, train_loader, val_loader, config)
    
    # Load best model
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    
    # Predict
    results = predict(model, dataset, config)
    results.to_csv('predictions_2025.csv', index=False)
    
    # Summary
    print_summary(results, config)


if __name__ == "__main__":
    main()
