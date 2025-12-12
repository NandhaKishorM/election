"""
Training Script for Kerala Election Prediction Model
"""

import os
import sys
import argparse
import time
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.dataset import create_mock_dataset, create_data_loaders
from models.election_predictor import ElectionPredictor, create_model
from utils.visualization import plot_training_history, plot_confusion_matrix


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """Training class for the election prediction model"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # TensorBoard writer
        log_dir = os.path.join(config.checkpoint_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)
    
    def train_epoch(self, train_loader) -> tuple:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Move data to device
            sentiment = batch['sentiment'].to(self.device)
            historical = batch['historical'].to(self.device)
            demographic = batch['demographic'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(sentiment, historical, demographic)
            
            # Compute loss
            loss = self.criterion(output['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(output['probs'], dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, val_loader) -> tuple:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        for batch in val_loader:
            sentiment = batch['sentiment'].to(self.device)
            historical = batch['historical'].to(self.device)
            demographic = batch['demographic'].to(self.device)
            labels = batch['label'].to(self.device)
            
            output = self.model(sentiment, historical, demographic)
            loss = self.criterion(output['logits'], labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(output['probs'], dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = None
    ) -> dict:
        """Full training loop"""
        epochs = epochs or self.config.epochs
        
        print(f"\n{'='*60}")
        print(f"Training Kerala Election Prediction Model")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, 'best_model.pt'),
                    epoch, val_loss, val_acc
                )
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final checkpoint
        self.save_checkpoint(
            os.path.join(self.config.checkpoint_dir, 'final_model.pt'),
            epochs, val_loss, val_acc
        )
        
        # Save training history
        self.save_history()
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        val_acc: float
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': {
                'sentiment_dim': 18,
                'historical_dim': self.config.historical_input_dim,
                'demographic_dim': self.config.num_demographic_features,
                'num_classes': self.config.num_classes,
                'parties': self.config.parties
            }
        }
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history"""
        # Save as JSON
        history_path = os.path.join(self.config.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save plot
        plot_path = os.path.join(self.config.checkpoint_dir, 'training_curves.png')
        plot_training_history(self.history, save_path=plot_path)
        print(f"Training curves saved to {plot_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Kerala Election Prediction Model')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mock_data', action='store_true', help='Use mock data for training')
    parser.add_argument('--num_booths', type=int, default=1000, help='Number of mock booths')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Configuration
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.mock_num_booths = args.num_booths
    
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Create dataset (using mock data for demonstration)
    print("\nCreating dataset...")
    dataset = create_mock_dataset(config, correlated=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimensions: {dataset.feature_dims}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, config,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Initialize trainer
    trainer = Trainer(model, config, device)
    
    # Train
    history = trainer.train(train_loader, val_loader, epochs=config.epochs)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    test_loss, test_acc, test_preds, test_labels = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save confusion matrix
    cm_path = os.path.join(config.checkpoint_dir, 'test_confusion_matrix.png')
    plot_confusion_matrix(
        test_labels, test_preds, config.parties,
        save_path=cm_path, normalize=True
    )
    print(f"Confusion matrix saved to {cm_path}")
    
    # Classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=config.parties))
    
    print("\nTraining complete! Model saved to:", config.checkpoint_dir)


if __name__ == "__main__":
    main()
