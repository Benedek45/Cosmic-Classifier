#!/usr/bin/env python3
"""
V10 Training Script - Temporal Sequences

Trains temporal models (LSTM/Transformer) on sequence data.
Standalone - no imports from v8/v9.

Usage:
    python run10/run_training_v10_temporal.py \\
        --dataset orbital_windows_dataset_v10_preprocessed \\
        --models temporal_lstm temporal_transformer \\
        --batch-size 512 \\
        --epochs 100 \\
        --workers 8
"""

import argparse
import sys
from pathlib import Path
import csv
import json

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import logging

# Add training directory to path
TRAINING_DIR = Path(__file__).resolve().parent / "training"
sys.path.insert(0, str(TRAINING_DIR))

from dataset_loader_v10 import create_dataloaders_v10
from temporal_models_v10 import get_temporal_model

# Enable Tensor Cores for better performance on modern GPUs
torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsLogger(Callback):
    """
    Custom PyTorch Lightning callback to log metrics to CSV and JSON files.
    Saves metrics after each validation epoch.
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory where metrics files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self.json_path = self.output_dir / "metrics.json"
        self.metrics_history = []

        # Create CSV header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_acc',
                    'val_loss', 'val_acc', 'val_auroc', 'learning_rate'
                ])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when validation epoch ends."""
        # Get current metrics from trainer
        metrics = trainer.callback_metrics

        # Extract metrics
        epoch = trainer.current_epoch
        train_loss = metrics.get('train_loss', None)
        train_acc = metrics.get('train_acc', None)
        val_loss = metrics.get('val_loss', None)
        val_acc = metrics.get('val_acc', None)
        val_auroc = metrics.get('val_auroc', None)

        # Get learning rate from optimizer
        learning_rate = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else None

        # Convert tensors to Python floats
        def to_float(val):
            if val is None:
                return None
            return float(val) if hasattr(val, 'item') else float(val)

        train_loss = to_float(train_loss)
        train_acc = to_float(train_acc)
        val_loss = to_float(val_loss)
        val_acc = to_float(val_acc)
        val_auroc = to_float(val_auroc)

        # Create metrics dict
        metrics_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auroc': val_auroc,
            'learning_rate': learning_rate
        }

        # Append to metrics history
        self.metrics_history.append(metrics_dict)

        # Write to CSV (append mode)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_auroc,
                learning_rate
            ])

        # Write to JSON (overwrite with full history)
        with open(self.json_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"Metrics saved to {self.csv_path} and {self.json_path}")


class TemporalTransitModule(pl.LightningModule):
    """
    PyTorch Lightning module for temporal transit detection.
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        super().__init__()

        self.save_hyperparameters()

        # Create model
        self.model = get_temporal_model(model_name)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Metrics - use torchmetrics (PyTorch Lightning 2.0+)
        try:
            from torchmetrics import Accuracy, AUROC
            self.train_acc = Accuracy(task='binary')
            self.val_acc = Accuracy(task='binary')
            self.val_auroc = AUROC(task='binary')
        except ImportError:
            # Fallback for older versions (PyTorch Lightning < 1.5)
            self.train_acc = pl.metrics.Accuracy(task='binary')
            self.val_acc = pl.metrics.Accuracy(task='binary')
            self.val_auroc = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)

        loss = self.criterion(logits, labels)

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        # Log
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch
        logits = self(sequences)

        loss = self.criterion(logits, labels)

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)

        # AUROC
        if self.val_auroc is not None:
            probs = torch.softmax(logits, dim=1)[:, 1]
            self.val_auroc(probs, labels)

        # Log
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

        if self.val_auroc is not None:
            self.log('val_auroc', self.val_auroc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc' if self.val_auroc is not None else 'val_acc',
                'interval': 'epoch'
            }
        }


def train_model(
    model_name: str,
    dataset_dir: str,
    output_dir: str,
    batch_size: int,
    epochs: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
    preload: bool,
    gpus: int
):
    """Train a single temporal model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*80}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders_v10(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        workers=workers,
        preload=preload
    )

    # Create module
    module = TemporalTransitModule(
        model_name=model_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Callbacks
    checkpoint_dir = Path(output_dir) / model_name / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = Path(output_dir) / model_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val_auroc:.4f}',
            monitor='val_auroc' if module.val_auroc is not None else 'val_acc',
            mode='max',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val_auroc' if module.val_auroc is not None else 'val_acc',
            mode='max',
            patience=10,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        MetricsLogger(output_dir=metrics_dir)
    ]

    # Logger
    log_dir = Path(output_dir) / model_name / "logs"
    logger_tb = TensorBoardLogger(
        save_dir=log_dir,
        name=model_name
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger_tb,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
        enable_progress_bar=True
    )

    # Train
    trainer.fit(module, train_loader, val_loader)

    logger.info(f"\n✅ Training complete for {model_name}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Logs: {log_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="V10 Training - Temporal Sequence Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to V10 preprocessed dataset"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["temporal_lstm"],
        choices=["temporal_lstm", "temporal_transformer"],
        help="Models to train (default: temporal_lstm)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="run10/runs_v10",
        help="Output directory for checkpoints/logs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)"
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload dataset into RAM"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate dataset
    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.exists():
        logger.error(f"❌ Dataset not found: {dataset_dir}")
        sys.exit(1)

    manifest = dataset_dir / "metadata" / "sequence_manifest_v10.csv"
    if not manifest.exists():
        logger.error(f"❌ Manifest not found: {manifest}")
        logger.error("This does not appear to be a V10 dataset.")
        logger.error("Run preprocessing first: python run_preprocessing_v10.py")
        sys.exit(1)

    # Print configuration
    logger.info("="*80)
    logger.info("V10 TEMPORAL SEQUENCE TRAINING")
    logger.info("="*80)
    logger.info(f"Dataset:       {dataset_dir}")
    logger.info(f"Models:        {', '.join(args.models)}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Batch size:    {args.batch_size}")
    logger.info(f"Epochs:        {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Workers:       {args.workers}")
    logger.info(f"Preload RAM:   {args.preload}")
    logger.info(f"GPUs:          {args.gpus}")
    logger.info("="*80)

    # Train each model
    for model_name in args.models:
        try:
            train_model(
                model_name=model_name,
                dataset_dir=str(dataset_dir),
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                epochs=args.epochs,
                workers=args.workers,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                preload=args.preload,
                gpus=args.gpus
            )
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n✅ All training complete!")
    logger.info(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
