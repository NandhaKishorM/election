"""
Historical Data Encoder Neural Network
Encodes past election data for prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal data.
    Adds position information to distinguish between different elections.
    """
    
    def __init__(self, d_model: int, max_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        Returns:
            Position-encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ElectionTransformerBlock(nn.Module):
    """
    Transformer block for processing election sequences.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (batch, seq_len, hidden_dim)
            mask: Optional attention mask
        Returns:
            output: Transformed output
            attn_weights: Attention weights
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attn_weights


class HistoricalEncoder(nn.Module):
    """
    Neural network to encode historical election data.
    
    Input: Historical features (batch, num_features) where features are
           organized as [election1_features, election2_features, election3_features]
    Output: Historical embedding (batch, hidden_dim)
    
    Architecture:
    - Reshape flat features into election sequence
    - Project each election to hidden dimension
    - Add positional encoding
    - Process with transformer blocks
    - Aggregate with attention pooling
    """
    
    def __init__(
        self,
        input_dim: int = 24,  # 3 elections * 8 features
        hidden_dim: int = 128,
        num_elections: int = 3,
        features_per_election: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_elections = num_elections
        self.features_per_election = features_per_election
        
        # Project election features to hidden dimension
        self.election_projection = nn.Sequential(
            nn.Linear(features_per_election, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for elections (temporal order matters)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_elections)
        
        # Transformer blocks for sequence processing
        self.transformer_blocks = nn.ModuleList([
            ElectionTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Attention pooling to aggregate sequence
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Trend encoder - specifically captures trend across elections
        self.trend_encoder = nn.LSTM(
            input_size=features_per_election,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Final projection combining transformer and LSTM outputs
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input historical features (batch, 24)
            return_attention: If True, also return attention weights
        
        Returns:
            embedding: Historical embedding (batch, hidden_dim)
            attention_weights (optional): Attention weights from transformer
        """
        batch_size = x.size(0)
        
        # Reshape to sequence: (batch, num_elections, features_per_election)
        x_seq = x.view(batch_size, self.num_elections, self.features_per_election)
        
        # Project each election to hidden dimension
        projected = self.election_projection(x_seq)  # (batch, num_elections, hidden_dim)
        
        # Add positional encoding
        projected = self.pos_encoding(projected)
        
        # Process through transformer blocks
        attention_weights = []
        hidden = projected
        for block in self.transformer_blocks:
            hidden, attn_w = block(hidden)
            attention_weights.append(attn_w)
        
        # Attention pooling over sequence
        pool_weights = self.pool_attention(hidden)  # (batch, num_elections, 1)
        pool_weights = F.softmax(pool_weights, dim=1)
        transformer_out = torch.sum(hidden * pool_weights, dim=1)  # (batch, hidden_dim)
        
        # LSTM for trend analysis
        lstm_out, (h_n, _) = self.trend_encoder(x_seq)
        # Concatenate forward and backward final hidden states
        lstm_agg = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, hidden_dim)
        
        # Combine transformer and LSTM outputs
        combined = torch.cat([transformer_out, lstm_agg], dim=-1)  # (batch, hidden_dim * 2)
        output = self.output_projection(combined)  # (batch, hidden_dim)
        
        if return_attention:
            return output, attention_weights
        
        return output
    
    def get_election_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get importance of each historical election for the prediction.
        
        Returns:
            importance: (batch, num_elections) how much each election contributes
        """
        batch_size = x.size(0)
        
        # Reshape and project
        x_seq = x.view(batch_size, self.num_elections, self.features_per_election)
        projected = self.election_projection(x_seq)
        projected = self.pos_encoding(projected)
        
        # Process through transformer
        hidden = projected
        for block in self.transformer_blocks:
            hidden, _ = block(hidden)
        
        # Get attention weights
        pool_weights = self.pool_attention(hidden).squeeze(-1)  # (batch, num_elections)
        importance = F.softmax(pool_weights, dim=1)
        
        return importance


class SwingAnalyzer(nn.Module):
    """
    Analyzes vote swing patterns between elections.
    """
    
    def __init__(
        self,
        num_parties: int = 4,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.num_parties = num_parties
        
        # Encode swing patterns
        self.swing_encoder = nn.Sequential(
            nn.Linear(num_parties * 2, hidden_dim),  # swing between 2 adjacent elections
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Aggregate swings
        self.swing_aggregator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
    
    def forward(self, vote_shares: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vote_shares: Vote shares over elections (batch, num_elections, num_parties)
        
        Returns:
            swing_embedding: Encoding of swing patterns (batch, hidden_dim)
        """
        batch_size = vote_shares.size(0)
        num_elections = vote_shares.size(1)
        
        # Compute swings between consecutive elections
        swings = []
        for i in range(num_elections - 1):
            current = vote_shares[:, i, :]
            next_el = vote_shares[:, i + 1, :]
            swing = torch.cat([next_el - current, next_el], dim=-1)  # change and new value
            swings.append(swing)
        
        swings = torch.stack(swings, dim=1)  # (batch, num_elections-1, num_parties*2)
        
        # Encode swings
        encoded_swings = self.swing_encoder(swings)  # (batch, num_elections-1, hidden_dim)
        
        # Aggregate with GRU
        _, h_n = self.swing_aggregator(encoded_swings)
        swing_embedding = h_n.squeeze(0)  # (batch, hidden_dim)
        
        return swing_embedding


# Testing
if __name__ == "__main__":
    # Test the encoder
    encoder = HistoricalEncoder(input_dim=24, hidden_dim=128)
    
    # Random input
    x = torch.randn(4, 24)
    
    # Forward pass
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get election importance
    importance = encoder.get_election_importance(x)
    print(f"Election importance shape: {importance.shape}")
    print(f"Sample importance: {importance[0]}")
    
    # With attention
    output, attn = encoder(x, return_attention=True)
    print(f"Number of attention weight tensors: {len(attn)}")
    
    # Parameter count
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {num_params:,}")
