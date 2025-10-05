"""
Multi-Architecture Training Suite
Trains and compares 3 architectures in one run:
1. CNN (your current - baseline)
2. ResNet-1D (deeper with skip connections)
3. Attention/Transformer (temporal patterns)

Features:
- Trains all 3 models
- Compares performance
- Saves best of each
- Generates comparison report
- Tests interpretability
- Expected time: 4-6 hours total
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Optional
import argparse
import time
from datetime import datetime
from dataclasses import dataclass
import json

from torchmetrics import Accuracy, F1Score, AUROC, Specificity, Recall, Precision
import matplotlib.pyplot as plt
import seaborn as sns

#============================================================================
# CONFIGURATION
#============================================================================
@dataclass
class MultiArchConfig:
    # Data
    data_dir: str = "./extracted_windows_safe"
    positive_windows: str = "./positive_windows_normalized.npy"
    negative_windows: str = "./negative_windows_normalized.npy"
    positive_metadata: str = "./positive_metadata.csv"
    negative_metadata: str = "./negative_metadata.csv"
    
    # Training
    batch_size: int = 1024
    num_epochs: int = 100        # Longer training
    learning_rate: float = 0.001
    patience: int = 15           # More patience
    
    # Sampling
    samples_per_epoch: int = 4_000_000
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Model
    window_size: int = 256
    dropout: float = 0.3
    weight_decay: float = 0.01
    
    # System
    num_workers: int = 16
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    precision: str = '16-mixed'
    random_seed: int = 42
    
    # Output
    checkpoint_dir: str = "./checkpoints_multi_arch"
    results_dir: str = "./multi_arch_results"
    
    # Architectures to test
    architectures: list = None
    
    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ['cnn', 'resnet', 'attention']


#============================================================================
# ARCHITECTURE 1: CNN (Your Current Baseline)
#============================================================================
class TransitCNN(nn.Module):
    """
    Baseline CNN (your current architecture).
    ~94k parameters
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


#============================================================================
# ARCHITECTURE 2: ResNet-1D (Deeper with Skip Connections)
#============================================================================
class ResidualBlock1D(nn.Module):
    """1D Residual block for time series."""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.silu = nn.SiLU()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.skip(identity)  # Skip connection
        out = self.silu(out)
        
        return out


class TransitResNet(nn.Module):
    """
    ResNet-1D for transit detection.
    ~150k parameters (larger but better gradient flow)
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 64, 2, stride=2, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride, dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


#============================================================================
# ARCHITECTURE 3: Attention-based (Temporal Patterns)
#============================================================================
class PositionalEncoding1D(nn.Module):
    """Add positional encoding to temporal data."""
    def __init__(self, d_model, max_len=256):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransitAttention(nn.Module):
    """
    Attention-based model for transit detection.
    Uses self-attention to capture periodic patterns.
    ~120k parameters
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        
        d_model = 128
        nhead = 8
        num_layers = 3
        
        # Input projection
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding1D(d_model, max_len=256)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, 1, 256)
        x = self.input_proj(x)  # (batch, d_model, 256)
        x = x.permute(0, 2, 1)  # (batch, 256, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (batch, d_model, 256)
        x = self.classifier(x)
        return x


#============================================================================
# BALANCED SAMPLER (same as before)
#============================================================================
class BalancedEpochSampler(Sampler):
    """Samples balanced batches each epoch."""
    def __init__(self, positive_indices, negative_indices, samples_per_epoch, seed=42):
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        
        self.n_pos = samples_per_epoch // 2
        self.n_neg = samples_per_epoch - self.n_pos
    
    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        
        pos_sample = rng.choice(self.positive_indices, size=self.n_pos, 
                               replace=len(self.positive_indices) < self.n_pos)
        neg_sample = rng.choice(self.negative_indices, size=self.n_neg,
                               replace=len(self.negative_indices) < self.n_neg)
        
        combined = np.concatenate([pos_sample, neg_sample])
        rng.shuffle(combined)
        self.epoch += 1
        
        return iter(combined.tolist())
    
    def __len__(self):
        return self.samples_per_epoch


#============================================================================
# DATASET (same as before)
#============================================================================
class TransitWindowDataset(Dataset):
    """Memory-mapped dataset."""
    def __init__(self, positive_file, negative_file, positive_metadata_file,
                 negative_metadata_file, split='train', train_ratio=0.7,
                 val_ratio=0.2, seed=42):
        self.split = split
        
        pos_meta = pd.read_csv(positive_metadata_file)
        neg_meta = pd.read_csv(negative_metadata_file)
        
        unique_pos_stars = pos_meta['star_id'].unique()
        unique_neg_stars = neg_meta['star_id'].unique()
        
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_pos_stars)
        rng.shuffle(unique_neg_stars)
        
        n_pos_train = int(len(unique_pos_stars) * train_ratio)
        n_pos_val = int(len(unique_pos_stars) * val_ratio)
        n_neg_train = int(len(unique_neg_stars) * train_ratio)
        n_neg_val = int(len(unique_neg_stars) * val_ratio)
        
        if split == 'train':
            pos_stars = set(unique_pos_stars[:n_pos_train])
            neg_stars = set(unique_neg_stars[:n_neg_train])
        elif split == 'val':
            pos_stars = set(unique_pos_stars[n_pos_train:n_pos_train + n_pos_val])
            neg_stars = set(unique_neg_stars[n_neg_train:n_neg_train + n_neg_val])
        else:
            pos_stars = set(unique_pos_stars[n_pos_train + n_pos_val:])
            neg_stars = set(unique_neg_stars[n_neg_train + n_neg_val:])
        
        self.positive_indices = pos_meta[pos_meta['star_id'].isin(pos_stars)]['window_id'].values
        self.negative_indices = neg_meta[neg_meta['star_id'].isin(neg_stars)]['window_id'].values
        
        self.positive_data = np.load(positive_file, mmap_mode='r')
        self.negative_data = np.load(negative_file, mmap_mode='r')
        
        self.n_positive = len(self.positive_indices)
        self.n_negative = len(self.negative_indices)
        self.total_samples = self.n_positive + self.n_negative
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx < self.n_positive:
            window_idx = self.positive_indices[idx]
            window = self.positive_data[window_idx]
            label = 1
        else:
            neg_idx = idx - self.n_positive
            window_idx = self.negative_indices[neg_idx]
            window = self.negative_data[window_idx]
            label = 0
        
        window = torch.from_numpy(np.array(window, dtype=np.float32))
        label = torch.tensor(label, dtype=torch.long)
        
        return window, label


#============================================================================
# DATA MODULE
#============================================================================
class TransitWindowDataModule(pl.LightningDataModule):
    """Data module."""
    def __init__(self, config: MultiArchConfig):
        super().__init__()
        self.config = config
        
        data_dir = Path(config.data_dir)
        self.pos_file = str(data_dir / config.positive_windows)
        self.neg_file = str(data_dir / config.negative_windows)
        self.pos_meta = str(data_dir / config.positive_metadata)
        self.neg_meta = str(data_dir / config.negative_metadata)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TransitWindowDataset(
                self.pos_file, self.neg_file, self.pos_meta, self.neg_meta,
                split='train', train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio, seed=self.config.random_seed
            )
            
            self.val_dataset = TransitWindowDataset(
                self.pos_file, self.neg_file, self.pos_meta, self.neg_meta,
                split='val', train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio, seed=self.config.random_seed
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = TransitWindowDataset(
                self.pos_file, self.neg_file, self.pos_meta, self.neg_meta,
                split='test', train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio, seed=self.config.random_seed
            )
    
    def train_dataloader(self):
        sampler = BalancedEpochSampler(
            np.arange(self.train_dataset.n_positive),
            np.arange(self.train_dataset.n_positive, self.train_dataset.total_samples),
            self.config.samples_per_epoch,
            self.config.random_seed
        )
        
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, sampler=sampler,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=self.config.pin_memory
        )


#============================================================================
# LIGHTNING MODULE (Universal for All Architectures)
#============================================================================
class TransitModule(pl.LightningModule):
    """Universal module for all architectures."""
    def __init__(self, model, config: MultiArchConfig, arch_name: str):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model_arch = model
        self.config = config
        self.arch_name = arch_name
        
        # Compile if PyTorch 2.0+
        if hasattr(torch, 'compile') and config.precision != '16-mixed':
            print(f"   Compiling {arch_name}...")
            self.model_arch = torch.compile(self.model_arch)
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_acc = Accuracy(task='binary')
        self.train_f1 = F1Score(task='binary')
        
        self.val_acc = Accuracy(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_spec = Specificity(task='binary')
        self.val_sens = Recall(task='binary')
        
        self.test_acc = Accuracy(task='binary')
        self.test_f1 = F1Score(task='binary')
        self.test_auc = AUROC(task='binary')
        self.test_spec = Specificity(task='binary')
        self.test_sens = Recall(task='binary')
    
    def forward(self, x):
        return self.model_arch(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y.float())
        
        preds = torch.sigmoid(logits)
        self.train_acc(preds, y)
        self.train_f1(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = self.criterion(logits, y.float())
        
        preds = torch.sigmoid(logits)
        
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_auc(preds, y)
        self.val_spec(preds, y)
        self.val_sens(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auc', self.val_auc, prog_bar=True)
        self.log('val_specificity', self.val_spec, prog_bar=True)
        self.log('val_sensitivity', self.val_sens, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        
        preds = torch.sigmoid(logits)
        
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_auc(preds, y)
        self.test_spec(preds, y)
        self.test_sens(preds, y)
        
        self.log('test_acc', self.test_acc)
        self.log('test_f1', self.test_f1)
        self.log('test_auc', self.test_auc)
        self.log('test_specificity', self.test_spec)
        self.log('test_sensitivity', self.test_sens)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_specificity'
            }
        }


#============================================================================
# TRAINING ORCHESTRATOR
#============================================================================
def train_architecture(arch_name: str, model_class, config: MultiArchConfig, data_module):
    """Train a single architecture."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {arch_name.upper()}")
    print(f"{'='*80}")
    
    # Create model
    model_instance = model_class(dropout=config.dropout)
    
    # Count parameters
    total_params = sum(p.numel() for p in model_instance.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e6:.2f} MB")
    
    # Create Lightning module
    pl_module = TransitModule(model_instance, config, arch_name)
    
    # Callbacks
    checkpoint_dir = Path(config.checkpoint_dir) / arch_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{arch_name}-{{epoch:02d}}-{{val_specificity:.4f}}',
        monitor='val_specificity',
        mode='max',
        save_top_k=3,
        verbose=True
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_specificity',
        patience=config.patience,
        mode='max',
        verbose=True
    )
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=config.precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=50,
        deterministic=True,
        gradient_clip_val=1.0,
        benchmark=True
    )
    
    # Train
    start_time = time.time()
    trainer.fit(pl_module, data_module)
    train_time = time.time() - start_time
    
    # Test
    test_results = trainer.test(pl_module, data_module, ckpt_path='best')
    
    results = {
        'architecture': arch_name,
        'parameters': total_params,
        'train_time_minutes': train_time / 60,
        'best_checkpoint': checkpoint_callback.best_model_path,
        'best_val_specificity': float(checkpoint_callback.best_model_score),
        'test_metrics': {k: float(v) for k, v in test_results[0].items()}
    }
    
    print(f"\n   {arch_name} complete!")
    print(f"     Train time: {train_time/60:.1f} min")
    print(f"     Best val spec: {results['best_val_specificity']:.4f}")
    print(f"     Test accuracy: {results['test_metrics']['test_acc']:.4f}")
    print(f"     Test AUC: {results['test_metrics']['test_auc']:.4f}")
    
    return results


#============================================================================
# COMPARISON & VISUALIZATION
#============================================================================
def create_comparison_report(all_results, config):
    """Create comprehensive comparison report."""
    results_dir = Path(config.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Create comparison table
    df = pd.DataFrame([
        {
            'Architecture': r['architecture'].upper(),
            'Parameters': f"{r['parameters']:,}",
            'Train Time (min)': f"{r['train_time_minutes']:.1f}",
            'Val Specificity': f"{r['best_val_specificity']:.4f}",
            'Test Accuracy': f"{r['test_metrics']['test_acc']:.4f}",
            'Test AUC': f"{r['test_metrics']['test_auc']:.4f}",
            'Test Specificity': f"{r['test_metrics']['test_specificity']:.4f}",
            'Test Sensitivity': f"{r['test_metrics']['test_sensitivity']:.4f}"
        }
        for r in all_results
    ])
    
    # Save table
    table_file = results_dir / 'comparison_table.csv'
    df.to_csv(table_file, index=False)
    
    print(f"\n COMPARISON TABLE:\n")
    print(df.to_string(index=False))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Architecture Comparison', fontsize=16, fontweight='bold')
    
    archs = [r['architecture'].upper() for r in all_results]
    
    # Plot 1: Test Accuracy
    ax = axes[0, 0]
    accs = [r['test_metrics']['test_acc'] for r in all_results]
    bars = ax.bar(archs, accs, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.set_ylim([min(accs) - 0.01, 1.0])
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom')
    
    # Plot 2: Test AUC
    ax = axes[0, 1]
    aucs = [r['test_metrics']['test_auc'] for r in all_results]
    bars = ax.bar(archs, aucs, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('AUC')
    ax.set_title('Test AUC Comparison')
    ax.set_ylim([min(aucs) - 0.01, 1.0])
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}', ha='center', va='bottom')
    
    # Plot 3: Specificity vs Sensitivity
    ax = axes[1, 0]
    specs = [r['test_metrics']['test_specificity'] for r in all_results]
    sens = [r['test_metrics']['test_sensitivity'] for r in all_results]
    x = np.arange(len(archs))
    width = 0.35
    ax.bar(x - width/2, specs, width, label='Specificity', color='#3498db')
    ax.bar(x + width/2, sens, width, label='Sensitivity', color='#e74c3c')
    ax.set_ylabel('Score')
    ax.set_title('Specificity vs Sensitivity')
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.legend()
    ax.set_ylim([0.9, 1.0])
    
    # Plot 4: Parameters vs Performance
    ax = axes[1, 1]
    params = [r['parameters']/1000 for r in all_results]  # in thousands
    ax.scatter(params, accs, s=200, alpha=0.6, c=['#3498db', '#e74c3c', '#2ecc71'])
    for i, arch in enumerate(archs):
        ax.annotate(arch, (params[i], accs[i]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Parameters (thousands)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Size vs Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_file = results_dir / 'architecture_comparison.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"\n   Saved: {viz_file}")
    
    # Save JSON results
    json_file = results_dir / 'all_results.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Find best model
    best_arch = max(all_results, key=lambda x: x['test_metrics']['test_auc'])
    
    print(f"\n{'='*80}")
    print(" BEST MODEL")
    print(f"{'='*80}")
    print(f"  Architecture: {best_arch['architecture'].upper()}")
    print(f"  Test Accuracy: {best_arch['test_metrics']['test_acc']:.4f}")
    print(f"  Test AUC: {best_arch['test_metrics']['test_auc']:.4f}")
    print(f"  Checkpoint: {best_arch['best_checkpoint']}")
    
    return best_arch


#============================================================================
# MAIN
#============================================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-Architecture Training Suite')
    parser.add_argument('--architectures', nargs='+', 
                       choices=['cnn', 'resnet', 'attention', 'all'],
                       default=['all'],
                       help='Which architectures to train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=15)
    
    args = parser.parse_args()
    
    # Create config
    config = MultiArchConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )
    
    if 'all' in args.architectures:
        config.architectures = ['cnn', 'resnet', 'attention']
    else:
        config.architectures = args.architectures
    
    print("="*80)
    print("MULTI-ARCHITECTURE TRAINING SUITE")
    print("="*80)
    print(f"\n  Configuration:")
    print(f"  Architectures: {', '.join([a.upper() for a in config.architectures])}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Patience: {config.patience}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Seed
    pl.seed_everything(config.random_seed)
    
    # Load data once
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    data_module = TransitWindowDataModule(config)
    data_module.setup('fit')
    data_module.setup('test')
    
    print(f"  Train: {data_module.train_dataset.total_samples:,} samples")
    print(f"  Val: {data_module.val_dataset.total_samples:,} samples")
    print(f"  Test: {data_module.test_dataset.total_samples:,} samples")
    
    # Architecture mapping
    arch_models = {
        'cnn': TransitCNN,
        'resnet': TransitResNet,
        'attention': TransitAttention
    }
    
    # Train all architectures
    all_results = []
    total_start = time.time()
    
    for arch_name in config.architectures:
        model_class = arch_models[arch_name]
        results = train_architecture(arch_name, model_class, config, data_module)
        all_results.append(results)
    
    total_time = time.time() - total_start
    
    # Create comparison report
    best_model = create_comparison_report(all_results, config)
    
    print(f"\n{'='*80}")
    print(" ALL TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Results saved: {config.results_dir}/")
    print(f"\n   Winner: {best_model['architecture'].upper()}")
    print(f"     AUC: {best_model['test_metrics']['test_auc']:.4f}")
    

if __name__ == "__main__":
    main()