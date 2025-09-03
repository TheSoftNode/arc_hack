"""
ARC Prize 2025 - Model Training and Management

This module handles training and management of neural network models
for the ARC reasoning pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC tasks.
    Converts ARC tasks into tensor format for neural network training.
    """
    
    def __init__(self, tasks: List[Any], max_grid_size: int = 30):
        self.tasks = tasks
        self.max_grid_size = max_grid_size
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[Dict[str, Any]]:
        """Convert tasks into training examples"""
        examples = []
        
        for task in self.tasks:
            for pair in task.train_pairs:
                input_grid = np.array(pair['input'])
                output_grid = np.array(pair['output'])
                
                # Pad grids to max size
                input_padded = self._pad_grid(input_grid)
                output_padded = self._pad_grid(output_grid)
                
                examples.append({
                    'input': torch.FloatTensor(input_padded),
                    'output': torch.FloatTensor(output_padded),
                    'task_id': task.task_id,
                    'original_input_shape': input_grid.shape,
                    'original_output_shape': output_grid.shape
                })
        
        return examples
    
    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """Pad grid to max_grid_size x max_grid_size"""
        h, w = grid.shape
        if h > self.max_grid_size or w > self.max_grid_size:
            # Crop if too large
            grid = grid[:self.max_grid_size, :self.max_grid_size]
            h, w = grid.shape
        
        # Pad to max size
        padded = np.zeros((self.max_grid_size, self.max_grid_size))
        padded[:h, :w] = grid
        
        return padded
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class ARCTransformer(nn.Module):
    """
    Transformer-based model for ARC reasoning.
    Treats grids as sequences and learns transformations.
    """
    
    def __init__(self, 
                 grid_size: int = 30,
                 vocab_size: int = 10,  # 10 colors in ARC
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.grid_size = grid_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embedding layers
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(grid_size * grid_size, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, grid_size, grid_size)
            
        Returns:
            Output tensor of shape (batch_size, grid_size, grid_size, vocab_size)
        """
        batch_size = x.shape[0]
        
        # Flatten grid to sequence: (batch_size, seq_len)
        x_flat = x.view(batch_size, -1).long()
        
        # Embed tokens: (batch_size, seq_len, d_model)
        x_embedded = self.input_embedding(x_flat)
        
        # Add position embeddings
        x_embedded = x_embedded + self.position_embedding.unsqueeze(0)
        
        # Apply transformer
        x_transformed = self.transformer_encoder(x_embedded)
        
        # Project to output vocabulary
        output_logits = self.output_projection(x_transformed)
        
        # Reshape back to grid: (batch_size, grid_size, grid_size, vocab_size)
        output_logits = output_logits.view(batch_size, self.grid_size, self.grid_size, self.vocab_size)
        
        return output_logits


class ModelTrainer:
    """Handles training of neural network models for ARC"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'auto',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Model trainer initialized on device: {self.device}")
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input'].to(self.device)
            targets = batch['output'].to(self.device).long()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Reshape for loss computation
            batch_size, h, w, vocab_size = outputs.shape
            outputs_flat = outputs.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Accuracy calculation
            predictions = outputs.argmax(dim=-1)
            correct = (predictions == targets).sum().item()
            correct_predictions += correct
            total_predictions += targets.numel()
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['output'].to(self.device).long()
                
                outputs = self.model(inputs)
                
                # Reshape for loss computation
                batch_size, h, w, vocab_size = outputs.shape
                outputs_flat = outputs.view(-1, vocab_size)
                targets_flat = targets.view(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
                
                # Accuracy
                predictions = outputs.argmax(dim=-1)
                correct = (predictions == targets).sum().item()
                correct_predictions += correct
                total_predictions += targets.numel()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def save_model(self, filepath: Path) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'class_name': self.model.__class__.__name__,
                # Add model-specific config here
            }
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")


def create_model_config(model_type: str = 'transformer') -> Dict[str, Any]:
    """Create default model configuration"""
    
    configs = {
        'transformer': {
            'grid_size': 30,
            'vocab_size': 10,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 512,
            'dropout': 0.1
        },
        # Add other model types here
    }
    
    return configs.get(model_type, configs['transformer'])
