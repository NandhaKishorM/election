"""
Election Predictor - Main Model
Multi-modal fusion model for predicting election outcomes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .sentiment_encoder import SentimentEncoder
from .historical_encoder import HistoricalEncoder


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between different feature types.
    Allows sentiment features to attend to historical features and vice versa.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Project query and key to same dimension
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, query_dim)
        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor (batch, query_dim) or (batch, seq_len, query_dim)
            key_value: Key/Value tensor (batch, key_dim) or (batch, seq_len, key_dim)
        
        Returns:
            output: Attended output (batch, query_dim)
            attn_weights: Attention weights
        """
        # Add sequence dimension if needed
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        
        # Project
        q = self.query_proj(query)
        k = self.key_proj(key_value)
        v = self.value_proj(key_value)
        
        # Attention
        attn_out, attn_weights = self.attention(q, k, v)
        
        # Project back and residual
        output = self.output_proj(attn_out)
        output = self.layer_norm(query + self.dropout(output))
        
        # Remove sequence dimension
        output = output.squeeze(1)
        
        return output, attn_weights


class DemographicEncoder(nn.Module):
    """
    Simple MLP encoder for demographic features.
    """
    
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ElectionPredictor(nn.Module):
    """
    Main election prediction model.
    
    Combines:
    - Sentiment features (social media analysis)
    - Historical features (past election data)
    - Demographic features (population characteristics)
    
    Architecture:
    1. Encode each modality with specialized encoders
    2. Apply cross-modal attention for feature interaction
    3. Fuse modalities with gated fusion
    4. Predict party probabilities with classification head
    """
    
    def __init__(
        self,
        sentiment_dim: int = 18,
        historical_dim: int = 24,
        demographic_dim: int = 12,
        sentiment_hidden: int = 256,
        historical_hidden: int = 128,
        demographic_hidden: int = 64,
        fusion_hidden: int = 256,
        num_classes: int = 4,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Individual modality encoders
        self.sentiment_encoder = SentimentEncoder(
            input_dim=sentiment_dim,
            hidden_dim=sentiment_hidden,
            dropout=dropout
        )
        
        self.historical_encoder = HistoricalEncoder(
            input_dim=historical_dim,
            hidden_dim=historical_hidden,
            dropout=dropout
        )
        
        self.demographic_encoder = DemographicEncoder(
            input_dim=demographic_dim,
            hidden_dim=demographic_hidden,
            dropout=dropout
        )
        
        # Cross-modal attention modules
        # Sentiment attends to historical
        self.sentiment_to_historical = CrossModalAttention(
            query_dim=sentiment_hidden,
            key_dim=historical_hidden,
            hidden_dim=fusion_hidden,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Historical attends to sentiment
        self.historical_to_sentiment = CrossModalAttention(
            query_dim=historical_hidden,
            key_dim=sentiment_hidden,
            hidden_dim=fusion_hidden,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Gated fusion layer
        total_hidden = sentiment_hidden + historical_hidden + demographic_hidden
        self.gate = nn.Sequential(
            nn.Linear(total_hidden, total_hidden),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(total_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden + total_hidden, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes)
        )
        
        # Uncertainty estimation head (for confidence scores)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        sentiment: torch.Tensor,
        historical: torch.Tensor,
        demographic: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            sentiment: Sentiment features (batch, 18)
            historical: Historical features (batch, 24)
            demographic: Demographic features (batch, 12)
            return_features: If True, return intermediate features
        
        Returns:
            Dictionary containing:
            - logits: Raw class scores (batch, num_classes)
            - probs: Class probabilities (batch, num_classes)
            - confidence: Prediction confidence (batch, 1)
            - features (optional): Intermediate features
        """
        # Encode each modality
        sentiment_emb = self.sentiment_encoder(sentiment)  # (batch, 256)
        historical_emb = self.historical_encoder(historical)  # (batch, 128)
        demographic_emb = self.demographic_encoder(demographic)  # (batch, 64)
        
        # Cross-modal attention
        sentiment_attended, sent_attn = self.sentiment_to_historical(
            sentiment_emb, historical_emb
        )
        historical_attended, hist_attn = self.historical_to_sentiment(
            historical_emb, sentiment_emb
        )
        
        # Concatenate all modalities
        concat_features = torch.cat([
            sentiment_attended,
            historical_attended,
            demographic_emb
        ], dim=-1)  # (batch, 256 + 128 + 64 = 448)
        
        # Gated fusion
        gate_weights = self.gate(concat_features)
        gated_features = concat_features * gate_weights
        fused = self.fusion(gated_features)  # (batch, 256)
        
        # Classification with skip connection
        classifier_input = torch.cat([fused, concat_features], dim=-1)
        logits = self.classifier(classifier_input)  # (batch, num_classes)
        
        # Probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Confidence estimation
        confidence = self.uncertainty_head(fused)  # (batch, 1)
        
        output = {
            'logits': logits,
            'probs': probs,
            'confidence': confidence
        }
        
        if return_features:
            output['features'] = {
                'sentiment': sentiment_emb,
                'historical': historical_emb,
                'demographic': demographic_emb,
                'fused': fused,
                'attention_weights': {
                    'sentiment_to_historical': sent_attn,
                    'historical_to_sentiment': hist_attn
                }
            }
        
        return output
    
    def predict(
        self,
        sentiment: torch.Tensor,
        historical: torch.Tensor,
        demographic: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions.
        
        Returns:
            predictions: Predicted class indices (batch,)
            probabilities: Class probabilities (batch, num_classes)
            confidence: Prediction confidence (batch,)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(sentiment, historical, demographic)
            predictions = torch.argmax(output['probs'], dim=-1)
            return predictions, output['probs'], output['confidence'].squeeze(-1)
    
    def get_feature_importance(
        self,
        sentiment: torch.Tensor,
        historical: torch.Tensor,
        demographic: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate importance of each modality.
        
        Uses gradient-based importance estimation.
        """
        # Enable gradients for inputs
        sentiment = sentiment.clone().requires_grad_(True)
        historical = historical.clone().requires_grad_(True)
        demographic = demographic.clone().requires_grad_(True)
        
        # Forward pass
        output = self.forward(sentiment, historical, demographic)
        
        # Get predicted class
        pred_class = torch.argmax(output['probs'], dim=-1)
        
        # Compute gradients for each sample's predicted class
        importance = {}
        
        # Sum of predicted class logits
        pred_logits = output['logits'].gather(1, pred_class.unsqueeze(1)).sum()
        
        # Backward pass
        pred_logits.backward()
        
        # Importance as absolute gradient magnitude
        importance['sentiment'] = sentiment.grad.abs().mean(dim=-1)
        importance['historical'] = historical.grad.abs().mean(dim=-1)
        importance['demographic'] = demographic.grad.abs().mean(dim=-1)
        
        # Normalize
        total = (importance['sentiment'] + importance['historical'] + 
                 importance['demographic'])
        for key in importance:
            importance[key] = importance[key] / (total + 1e-8)
        
        return importance


def create_model(config) -> ElectionPredictor:
    """Create model from config"""
    return ElectionPredictor(
        sentiment_dim=18,  # 4 * 4 + 2
        historical_dim=config.historical_input_dim,
        demographic_dim=config.num_demographic_features,
        sentiment_hidden=config.sentiment_hidden_dim,
        historical_hidden=config.historical_hidden_dim,
        demographic_hidden=config.demographic_hidden_dim,
        fusion_hidden=config.fusion_hidden_dim,
        num_classes=config.num_classes,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout
    )


def load_model(checkpoint_path: str, config=None) -> ElectionPredictor:
    """Load model from checkpoint"""
    if config is None:
        from config import Config
        config = Config()
    
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


# Testing
if __name__ == "__main__":
    # Test the model
    model = ElectionPredictor()
    
    # Random inputs
    batch_size = 4
    sentiment = torch.randn(batch_size, 18)
    historical = torch.randn(batch_size, 24)
    demographic = torch.randn(batch_size, 12)
    
    # Forward pass
    output = model(sentiment, historical, demographic, return_features=True)
    
    print("Output keys:", output.keys())
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Confidence shape: {output['confidence'].shape}")
    
    # Prediction
    preds, probs, conf = model.predict(sentiment, historical, demographic)
    print(f"\nPredictions: {preds}")
    print(f"Probabilities:\n{probs}")
    print(f"Confidence: {conf}")
    
    # Feature importance
    model.train()
    importance = model.get_feature_importance(sentiment, historical, demographic)
    print(f"\nFeature importance:")
    for key, val in importance.items():
        print(f"  {key}: {val.mean().item():.4f}")
    
    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
