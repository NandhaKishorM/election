"""
Historical Election Data Loader
Loads and processes historical booth-level election data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class BoothResult:
    """Single booth election result"""
    booth_id: str
    ward_id: str
    district: str
    election_year: int
    winner_party: str
    vote_shares: Dict[str, float]  # party -> vote share
    voter_turnout: float
    total_votes: int
    valid_votes: int
    rejected_votes: int
    num_candidates: int
    margin: float  # victory margin percentage


class HistoricalDataLoader:
    """
    Load historical election data.
    In production, this would load from CSV files or databases
    containing actual Kerala local body election results.
    """
    
    def __init__(self, config, data_path: Optional[str] = None):
        self.config = config
        self.data_path = data_path or config.data_dir
    
    def load_election_data(self, year: int) -> pd.DataFrame:
        """Load election data for a specific year"""
        file_path = os.path.join(self.data_path, f"election_{year}.csv")
        
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"Warning: No data file found at {file_path}")
            return None
    
    def load_all_historical_data(self) -> Dict[int, pd.DataFrame]:
        """Load all available historical data"""
        years = [2010, 2015, 2020]  # Kerala local body election years
        data = {}
        
        for year in years:
            df = self.load_election_data(year)
            if df is not None:
                data[year] = df
        
        return data
    
    def compute_features(
        self,
        booth_id: str,
        historical_data: Dict[int, pd.DataFrame]
    ) -> np.ndarray:
        """
        Compute historical features for a single booth.
        
        Features per election:
        - vote_share per party (4)
        - turnout (1)
        - margin (1)
        - winner encoded (4 one-hot)
        
        Total: 10 features per election * 3 elections = 30 features
        """
        num_features = 10 * len(historical_data)
        features = np.zeros(num_features)
        
        for i, (year, df) in enumerate(sorted(historical_data.items())):
            booth_data = df[df['booth_id'] == booth_id]
            if len(booth_data) == 0:
                continue
            
            row = booth_data.iloc[0]
            offset = i * 10
            
            # Vote shares for each party
            for j, party in enumerate(self.config.parties):
                col = f'{party}_vote_share'
                if col in row:
                    features[offset + j] = row[col]
            
            # Turnout
            if 'turnout' in row:
                features[offset + 4] = row['turnout']
            
            # Victory margin
            if 'margin' in row:
                features[offset + 5] = row['margin']
            
            # Winner one-hot encoding
            if 'winner' in row:
                winner_idx = self.config.parties.index(row['winner'])
                features[offset + 6 + winner_idx] = 1.0
        
        return features


class MockHistoricalGenerator:
    """
    Generate mock historical election data for demonstration.
    Simulates Kerala local body election patterns.
    """
    
    def __init__(self, config):
        self.config = config
        self.parties = config.parties
        self.districts = config.districts
    
    def generate_booth_history(
        self,
        num_booths: int = 1000,
        num_elections: int = 3
    ) -> np.ndarray:
        """
        Generate mock historical features for booths.
        
        Features per booth (for each of 3 elections):
        - vote_share per party (4)
        - turnout
        - margin
        - swing from previous
        - incumbency factor
        Total: 8 features per election * 3 = 24 features
        """
        np.random.seed(44)
        
        num_features = self.config.historical_input_dim  # 24
        history = np.zeros((num_booths, num_features))
        
        for i in range(num_booths):
            # Generate booth characteristic (some booths strongly favor one party)
            booth_lean = np.random.dirichlet(np.ones(4) * 0.5)
            
            prev_winner = None
            prev_vote_shares = None
            
            for e in range(num_elections):
                offset = e * 8
                
                # Vote shares (must sum to 1)
                if prev_vote_shares is None:
                    noise = np.random.normal(0, 0.1, 4)
                    vote_shares = np.clip(booth_lean + noise, 0.01, 1)
                else:
                    # Add swing from previous election
                    swing = np.random.normal(0, 0.05, 4)
                    vote_shares = np.clip(prev_vote_shares + swing, 0.01, 1)
                
                vote_shares = vote_shares / vote_shares.sum()
                history[i, offset:offset+4] = vote_shares
                
                # Turnout (typically 70-85% in Kerala)
                history[i, offset+4] = np.random.beta(8, 2) * 0.3 + 0.6
                
                # Margin (winner's advantage)
                sorted_shares = np.sort(vote_shares)[::-1]
                history[i, offset+5] = sorted_shares[0] - sorted_shares[1]
                
                # Swing from previous
                if prev_vote_shares is not None:
                    max_swing = np.max(np.abs(vote_shares - prev_vote_shares))
                    history[i, offset+6] = max_swing
                else:
                    history[i, offset+6] = 0.0
                
                # Incumbency factor (1 if same party won consecutively)
                current_winner = np.argmax(vote_shares)
                if prev_winner is not None and current_winner == prev_winner:
                    history[i, offset+7] = 1.0
                else:
                    history[i, offset+7] = 0.0
                
                prev_winner = current_winner
                prev_vote_shares = vote_shares.copy()
        
        return history.astype(np.float32)
    
    def generate_with_correlation_to_labels(
        self,
        num_booths: int = 1000,
        labels: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate historical data correlated with actual labels.
        This simulates realistic pattern where history predicts outcome.
        """
        history = self.generate_booth_history(num_booths)
        
        if labels is not None:
            # Adjust the most recent election's vote shares to correlate with labels
            for i in range(num_booths):
                winner_party = labels[i]
                
                # Last election is at offset 16 (3rd election)
                last_offset = 16
                
                # Boost winner's historical vote share
                boost = np.random.uniform(0.05, 0.15)
                history[i, last_offset + winner_party] += boost
                
                # Reduce others proportionally
                for j in range(4):
                    if j != winner_party:
                        history[i, last_offset + j] -= boost / 3
                
                # Renormalize
                vote_shares = history[i, last_offset:last_offset+4]
                vote_shares = np.clip(vote_shares, 0.01, 1)
                history[i, last_offset:last_offset+4] = vote_shares / vote_shares.sum()
        
        return history


def get_historical_feature_names() -> List[str]:
    """Get names of historical features"""
    from config import Config
    config = Config()
    
    elections = ['2010', '2015', '2020']
    feature_names = []
    
    for year in elections:
        for party in config.parties:
            feature_names.append(f"{year}_{party}_vote_share")
        feature_names.extend([
            f"{year}_turnout",
            f"{year}_margin",
            f"{year}_swing",
            f"{year}_incumbency"
        ])
    
    return feature_names
