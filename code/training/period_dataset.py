"""
Dataset Loader for Period Detection Training

Loads pre-generated period training data from HDF5 files.
Handles variable-length lightcurves with padding/collation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PeriodDetectionDataset(Dataset):
    """
    Dataset for period detection training.

    Loads pre-generated data containing:
    - Full lightcurve (flux + time)
    - Model 1 transit probability timeline
    - Ground truth period
    """

    def __init__(
        self,
        h5_file: Path,
        split: str = 'train',
        train_fraction: float = 0.8,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            h5_file: Path to HDF5 file with pre-generated data
            split: 'train' or 'val'
            train_fraction: Fraction for training (default 0.8)
            max_samples: Limit number of samples (for debugging)
        """
        self.h5_file = Path(h5_file)

        if not self.h5_file.exists():
            raise FileNotFoundError(f"Dataset not found: {h5_file}")

        # Load metadata
        with h5py.File(h5_file, 'r') as f:
            self.num_samples = f.attrs['num_samples']

        logger.info(f"Loading period dataset from: {h5_file}")
        logger.info(f"  Total samples: {self.num_samples}")

        # Limit samples if requested
        if max_samples is not None:
            self.num_samples = min(self.num_samples, max_samples)
            logger.info(f"  Limited to: {self.num_samples}")

        # Train/val split
        indices = np.random.RandomState(42).permutation(self.num_samples)
        split_idx = int(self.num_samples * train_fraction)

        if split == 'train':
            self.indices = indices[:split_idx]
        elif split == 'val':
            self.indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")

        logger.info(f"  {split.capitalize()} samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Dict:
        sample_idx = self.indices[idx]

        with h5py.File(self.h5_file, 'r') as f:
            grp = f[f'sample_{sample_idx:06d}']

            time = grp['time'][:]
            flux = grp['flux'][:]
            timeline = grp['timeline'][:]
            period = grp.attrs['period']
            target_id = grp.attrs['target_id']
            length = grp.attrs['num_samples']

        return {
            'flux': torch.FloatTensor(flux),
            'timeline': torch.FloatTensor(timeline),
            'time': torch.FloatTensor(time),
            'period': torch.FloatTensor([period]),
            'length': length,
            'target_id': target_id
        }


def collate_fn(batch) -> Dict:
    """
    Collate variable-length sequences with padding.

    Args:
        batch: List of samples from PeriodDetectionDataset

    Returns:
        Batched tensors with padding
    """
    # Find max length in batch
    max_len = max(b['length'] for b in batch)
    batch_size = len(batch)

    # Pad all to max length
    flux_padded = torch.zeros(batch_size, max_len)
    timeline_padded = torch.zeros(batch_size, max_len)
    time_padded = torch.zeros(batch_size, max_len)
    periods = torch.zeros(batch_size, 1)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    target_ids = []

    for i, b in enumerate(batch):
        L = b['length']
        flux_padded[i, :L] = b['flux']
        timeline_padded[i, :L] = b['timeline']
        time_padded[i, :L] = b['time']
        periods[i] = b['period']
        lengths[i] = L
        target_ids.append(b['target_id'])

    return {
        'flux': flux_padded,
        'timeline': timeline_padded,
        'time': time_padded,
        'period': periods,
        'length': lengths,
        'target_id': target_ids
    }


def create_period_dataloaders(
    h5_file: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_fraction: float = 0.8,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for period detection.

    Args:
        h5_file: Path to HDF5 dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        train_fraction: Train/val split fraction
        max_samples: Limit dataset size (for debugging)

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = PeriodDetectionDataset(
        h5_file,
        split='train',
        train_fraction=train_fraction,
        max_samples=max_samples
    )

    val_dataset = PeriodDetectionDataset(
        h5_file,
        split='val',
        train_fraction=train_fraction,
        max_samples=max_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logger.info(f"Created dataloaders:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python period_dataset.py <path_to_h5_file>")
        sys.exit(1)

    h5_file = Path(sys.argv[1])

    print(f"Testing dataset loading from: {h5_file}")

    # Create dataset
    dataset = PeriodDetectionDataset(h5_file, split='train', max_samples=10)

    print(f"\nDataset size: {len(dataset)}")

    # Load first sample
    sample = dataset[0]

    print(f"\nFirst sample:")
    print(f"  Flux shape: {sample['flux'].shape}")
    print(f"  Timeline shape: {sample['timeline'].shape}")
    print(f"  Period: {sample['period'].item():.3f} days")
    print(f"  Length: {sample['length']}")
    print(f"  Target ID: {sample['target_id']}")

    # Test dataloader
    train_loader, val_loader = create_period_dataloaders(
        h5_file,
        batch_size=4,
        max_samples=20
    )

    print(f"\nTesting dataloader...")
    batch = next(iter(train_loader))

    print(f"  Batch flux shape: {batch['flux'].shape}")
    print(f"  Batch periods: {batch['period'].squeeze().tolist()}")
    print(f"  Batch lengths: {batch['length'].tolist()}")

    print("\n✅ Dataset loading works!")
