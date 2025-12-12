"""
Inference Script for Kerala Election Prediction
Load trained model and make predictions on new data
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.dataset import create_mock_dataset, ElectionDataset
from models.election_predictor import load_model, create_model
from utils.visualization import (
    plot_prediction_distribution,
    plot_district_predictions,
    create_prediction_report
)


class ElectionPredictor:
    """
    Inference class for election prediction.
    Handles model loading, prediction, and result export.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Config = None,
        device: str = 'auto'
    ):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration object
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.config = config or Config()
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Party names
        self.parties = self.config.parties
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load model from checkpoint"""
        print(f"Loading model from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with saved config if available
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            model = create_model(self.config)
        else:
            model = create_model(self.config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully!")
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")
        
        return model
    
    @torch.no_grad()
    def predict(
        self,
        sentiment: torch.Tensor,
        historical: torch.Tensor,
        demographic: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions on input data.
        
        Args:
            sentiment: Sentiment features (batch, 18)
            historical: Historical features (batch, 24)
            demographic: Demographic features (batch, 12)
        
        Returns:
            Dictionary with predictions, probabilities, and confidence
        """
        # Move to device
        sentiment = sentiment.to(self.device)
        historical = historical.to(self.device)
        demographic = demographic.to(self.device)
        
        # Forward pass
        output = self.model(sentiment, historical, demographic)
        
        # Get predictions
        probs = output['probs'].cpu().numpy()
        predictions = np.argmax(probs, axis=1)
        confidence = output['confidence'].cpu().numpy().squeeze()
        
        return {
            'predictions': predictions,
            'probabilities': probs,
            'confidence': confidence,
            'predicted_parties': [self.parties[p] for p in predictions]
        }
    
    def predict_dataset(
        self,
        dataset: ElectionDataset,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Make predictions on entire dataset.
        
        Args:
            dataset: ElectionDataset to predict on
            batch_size: Batch size for inference
        
        Returns:
            DataFrame with predictions for each booth
        """
        from torch.utils.data import DataLoader
        from data.dataset import collate_fn
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        all_results = []
        
        print("Making predictions...")
        for batch in tqdm(loader):
            result = self.predict(
                batch['sentiment'],
                batch['historical'],
                batch['demographic']
            )
            
            for i in range(len(batch['booth_id'])):
                all_results.append({
                    'booth_id': batch['booth_id'][i],
                    'ward_id': batch['ward_id'][i],
                    'prediction': result['predictions'][i],
                    'predicted_party': result['predicted_parties'][i],
                    'confidence': result['confidence'][i] if np.ndim(result['confidence']) > 0 else result['confidence'],
                    'LDF_prob': result['probabilities'][i, 0],
                    'UDF_prob': result['probabilities'][i, 1],
                    'NDA_prob': result['probabilities'][i, 2],
                    'OTHERS_prob': result['probabilities'][i, 3]
                })
        
        return pd.DataFrame(all_results)
    
    def get_feature_importance(
        self,
        sentiment: torch.Tensor,
        historical: torch.Tensor,
        demographic: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get modality importance for predictions.
        
        Returns average importance across samples.
        """
        self.model.train()  # Enable gradients
        
        importance = self.model.get_feature_importance(
            sentiment.to(self.device),
            historical.to(self.device),
            demographic.to(self.device)
        )
        
        self.model.eval()
        
        return {
            'Sentiment': importance['sentiment'].mean().item(),
            'Historical': importance['historical'].mean().item(),
            'Demographic': importance['demographic'].mean().item()
        }
    
    def generate_report(
        self,
        predictions_df: pd.DataFrame,
        true_labels: Optional[np.ndarray] = None,
        output_dir: str = 'predictions'
    ) -> str:
        """
        Generate comprehensive prediction report.
        
        Args:
            predictions_df: DataFrame with predictions
            true_labels: Optional true labels for evaluation
            output_dir: Directory to save report
        
        Returns:
            Path to report directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions CSV
        csv_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
        
        # Summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_booths': len(predictions_df),
            'predictions_by_party': predictions_df['predicted_party'].value_counts().to_dict(),
            'average_confidence': float(predictions_df['confidence'].mean()),
            'high_confidence_booths': int((predictions_df['confidence'] > 0.7).sum()),
            'low_confidence_booths': int((predictions_df['confidence'] < 0.5).sum())
        }
        
        # Add accuracy if true labels provided
        if true_labels is not None:
            accuracy = (predictions_df['prediction'].values == true_labels).mean()
            summary['accuracy'] = float(accuracy)
        
        # Save summary JSON
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualizations
        probs = predictions_df[['LDF_prob', 'UDF_prob', 'NDA_prob', 'OTHERS_prob']].values
        
        # Probability distribution
        plot_prediction_distribution(
            probs,
            self.parties,
            save_path=os.path.join(output_dir, 'probability_distribution.png')
        )
        
        # Print summary
        print("\n" + "="*60)
        print("KERALA ELECTION PREDICTION SUMMARY")
        print("="*60)
        print(f"Total Booths Predicted: {summary['total_booths']}")
        print(f"\nPredicted Winners by Party:")
        for party, count in summary['predictions_by_party'].items():
            pct = count / summary['total_booths'] * 100
            print(f"  {party}: {count} ({pct:.1f}%)")
        print(f"\nAverage Confidence: {summary['average_confidence']:.2%}")
        print(f"High Confidence (>70%): {summary['high_confidence_booths']} booths")
        print(f"Low Confidence (<50%): {summary['low_confidence_booths']} booths")
        
        if 'accuracy' in summary:
            print(f"\nTest Accuracy: {summary['accuracy']:.2%}")
        
        print("="*60)
        print(f"\nFull report saved to: {output_dir}")
        
        return output_dir


def predict_tomorrow(predictor: ElectionPredictor, config: Config) -> pd.DataFrame:
    """
    Predict tomorrow's election results.
    
    This function would typically:
    1. Fetch latest social media sentiment data
    2. Load historical election data
    3. Load demographic features
    4. Make predictions
    
    For demonstration, we use mock data.
    """
    print("\n" + "="*60)
    print("PREDICTING TOMORROW'S KERALA LOCAL BODY ELECTION")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Prediction for: {(datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}")
    print("="*60 + "\n")
    
    # In production, this would fetch real data
    # For demonstration, we create mock data
    print("Loading data...")
    
    # Create dataset with mock data
    dataset = create_mock_dataset(config, correlated=True)
    
    # Make predictions
    predictions_df = predictor.predict_dataset(dataset)
    
    # Add district information (from dataset)
    if hasattr(dataset, 'districts') and dataset.districts:
        predictions_df['district'] = dataset.districts
    else:
        # Assign districts based on booth index
        num_booths = len(predictions_df)
        booths_per_district = num_booths // len(config.districts)
        predictions_df['district'] = [
            config.districts[min(i // booths_per_district, len(config.districts) - 1)]
            for i in range(num_booths)
        ]
    
    # Generate district-wise summary
    print("\n" + "-"*60)
    print("DISTRICT-WISE PREDICTIONS")
    print("-"*60)
    
    district_summary = predictions_df.groupby('district')['predicted_party'].value_counts().unstack(fill_value=0)
    print(district_summary.to_string())
    
    # Overall winner projection
    print("\n" + "-"*60)
    print("OVERALL PROJECTION")
    print("-"*60)
    
    total_by_party = predictions_df['predicted_party'].value_counts()
    total = len(predictions_df)
    
    for party in config.parties:
        if party in total_by_party:
            count = total_by_party[party]
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"{party:8s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Predicted winner
    winner = total_by_party.idxmax()
    winner_seats = total_by_party.max()
    print(f"\n🏆 PROJECTED WINNER: {winner} with {winner_seats} seats ({winner_seats/total*100:.1f}%)")
    
    return predictions_df


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kerala Election Prediction Inference')
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='predictions',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--mock_data',
        action='store_true',
        help='Use mock data for inference'
    )
    parser.add_argument(
        '--num_booths',
        type=int,
        default=1000,
        help='Number of booths to predict (for mock data)'
    )
    parser.add_argument(
        '--predict_tomorrow',
        action='store_true',
        help='Generate prediction for tomorrow\'s election'
    )
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Configuration
    config = Config()
    config.mock_num_booths = args.num_booths
    
    # Check if checkpoint exists
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try relative to script directory
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            args.checkpoint
        )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first using: python train.py")
        
        # For demonstration, create and train a quick model
        print("\nCreating and training a demonstration model...")
        
        # Quick training for demo
        from data.dataset import create_mock_dataset, create_data_loaders
        from models.election_predictor import create_model
        import torch.optim as optim
        
        dataset = create_mock_dataset(config, correlated=True)
        train_loader, val_loader, _ = create_data_loaders(dataset, config)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_model(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        print("Quick training (5 epochs)...")
        model.train()
        for epoch in range(5):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(
                    batch['sentiment'].to(device),
                    batch['historical'].to(device),
                    batch['demographic'].to(device)
                )
                loss = criterion(output['logits'], batch['label'].to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/5 - Loss: {total_loss/len(train_loader):.4f}")
        
        # Save quick model
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
        torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
        print(f"Demo model saved to {checkpoint_path}\n")
    
    # Initialize predictor
    predictor = ElectionPredictor(checkpoint_path, config, args.device)
    
    if args.predict_tomorrow:
        # Predict tomorrow's election
        predictions_df = predict_tomorrow(predictor, config)
    else:
        # Standard inference on dataset
        print("\nLoading dataset...")
        dataset = create_mock_dataset(config, correlated=True)
        
        # Make predictions
        predictions_df = predictor.predict_dataset(dataset)
        
        # Add district info
        num_booths = len(predictions_df)
        booths_per_district = num_booths // len(config.districts)
        predictions_df['district'] = [
            config.districts[min(i // booths_per_district, len(config.districts) - 1)]
            for i in range(num_booths)
        ]
    
    # Generate report
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.output_dir
    )
    
    # Get true labels for evaluation (if using mock data)
    true_labels = dataset.labels.numpy() if hasattr(dataset, 'labels') else None
    
    predictor.generate_report(
        predictions_df,
        true_labels=true_labels,
        output_dir=output_dir
    )
    
    # Get feature importance
    sample = dataset[0]
    importance = predictor.get_feature_importance(
        sample['sentiment'].unsqueeze(0),
        sample['historical'].unsqueeze(0),
        sample['demographic'].unsqueeze(0)
    )
    
    print("\nFeature Modality Importance:")
    for modality, imp in importance.items():
        bar = "█" * int(imp * 50)
        print(f"  {modality:12s}: {imp:5.1%} {bar}")


if __name__ == "__main__":
    main()
