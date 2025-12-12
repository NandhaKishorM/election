"""
Sentiment Encoder Neural Network
Encodes social media sentiment features for election prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for weighting sentiment over time.
    More recent sentiments typically have higher importance.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            mask: Optional attention mask
        
        Returns:
            output: Attended output (batch, seq_len, hidden_dim)
            attn_weights: Attention weights (batch, seq_len, seq_len)
        """
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=mask)
        output = self.layer_norm(x + attn_output)
        return output, attn_weights


class SentimentEncoder(nn.Module):
    """
    Neural network to encode sentiment features into a fixed-size embedding.
    
    Input: Sentiment features (batch, num_sentiment_features)
    Output: Sentiment embedding (batch, hidden_dim)
    
    Architecture:
    - Input projection layer
    - Party-wise feature attention
    - Temporal modeling with LSTM
    - Output projection
    """
    
    def __init__(
        self,
        input_dim: int = 18,  # 4 parties * 4 features + 2 global
        hidden_dim: int = 256,
        num_parties: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_parties = num_parties
        self.features_per_party = 4  # avg_sentiment, std, mentions, trend
        
        # Input projection - expand features for better representation
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Party-specific encoders
        # Each party's features get their own small encoder
        self.party_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.features_per_party, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, hidden_dim // 4)
            )
            for _ in range(num_parties)
        ])
        
        # Global feature encoder (total_engagement, volatility)
        self.global_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Cross-party attention
        # Allows model to compare sentiments across parties
        self.cross_party_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 4,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion layer - combines party-specific and global features
        # 4 parties * (hidden_dim//4) + global (hidden_dim//4) = 5 * hidden_dim//4
        fusion_input_dim = (num_parties + 1) * (hidden_dim // 4)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat input_proj and fusion
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_party_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input sentiment features (batch, 18)
            return_party_embeddings: If True, also return per-party embeddings
        
        Returns:
            embedding: Sentiment embedding (batch, hidden_dim)
            party_embeddings (optional): Per-party embeddings (batch, num_parties, hidden_dim//4)
        """
        batch_size = x.size(0)
        
        # Overall input projection
        input_encoded = self.input_projection(x)  # (batch, hidden_dim)
        
        # Split input into party-specific and global features
        party_features = []
        for i in range(self.num_parties):
            start_idx = i * self.features_per_party
            end_idx = start_idx + self.features_per_party
            party_feat = x[:, start_idx:end_idx]  # (batch, 4)
            party_features.append(party_feat)
        
        global_features = x[:, -2:]  # Last 2 features (batch, 2)
        
        # Encode each party's features
        party_embeddings = []
        for i, (encoder, feat) in enumerate(zip(self.party_encoders, party_features)):
            party_emb = encoder(feat)  # (batch, hidden_dim//4)
            party_embeddings.append(party_emb)
        
        party_embeddings = torch.stack(party_embeddings, dim=1)  # (batch, num_parties, hidden_dim//4)
        
        # Cross-party attention
        attended_parties, _ = self.cross_party_attention(
            party_embeddings, party_embeddings, party_embeddings
        )  # (batch, num_parties, hidden_dim//4)
        
        # Encode global features
        global_emb = self.global_encoder(global_features)  # (batch, hidden_dim//4)
        
        # Fusion - concatenate all embeddings
        party_flat = attended_parties.reshape(batch_size, -1)  # (batch, num_parties * hidden_dim//4)
        fusion_input = torch.cat([party_flat, global_emb], dim=-1)  # (batch, (num_parties+1) * hidden_dim//4)
        
        fused = self.fusion(fusion_input)  # (batch, hidden_dim)
        
        # Combine with input projection via residual-style connection
        combined = torch.cat([input_encoded, fused], dim=-1)  # (batch, hidden_dim * 2)
        output = self.output_projection(combined)  # (batch, hidden_dim)
        
        if return_party_embeddings:
            return output, attended_parties
        
        return output
    
    def get_party_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get importance scores for each party based on sentiment.
        
        Returns:
            importance: (batch, num_parties) normalized importance scores
        """
        _, party_embeddings = self.forward(x, return_party_embeddings=True)
        
        # Use L2 norm as importance measure
        importance = torch.norm(party_embeddings, p=2, dim=-1)  # (batch, num_parties)
        importance = F.softmax(importance, dim=-1)  # Normalize
        
        return importance


# Testing
if __name__ == "__main__":
    # Test the encoder
    encoder = SentimentEncoder(input_dim=18, hidden_dim=256)
    
    # Random input
    x = torch.randn(4, 18)
    
    # Forward pass
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get party importance
    importance = encoder.get_party_importance(x)
    print(f"Party importance shape: {importance.shape}")
    print(f"Sample importance: {importance[0]}")
    
    # Parameter count
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {num_params:,}")
