"""
Dataset Loader V10 - Temporal Sequences

Loads temporal sequences for training.
Standalone - no imports from v8/v9.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class TemporalSequenceDataset(Dataset):
    """
    Dataset loader for V10 temporal sequences.

    Returns:
        sequence: (5, 256) - 5 windows per sample
        label: 0 or 1 (binary: 0=negative, 1=transit)
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = 'train',
        preload_into_ram: bool = False,
        train_fraction: float = 0.8
    ):
        """
        Args:
            dataset_dir: Path to V10 dataset
            split: 'train' or 'val'
            preload_into_ram: Load all chunks into RAM
            train_fraction: Fraction for training (default 0.8)
        """
        self.dataset_dir = Path(dataset_dir)
        self.chunks_dir = self.dataset_dir / "chunks"
        self.manifest_path = self.dataset_dir / "metadata" / "sequence_manifest_v10.csv"

        # Load manifest
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.manifest = pd.read_csv(self.manifest_path)

        # Convert labels: 1 → 0 (negative), 2 → 1 (transit)
        self.manifest['label'] = self.manifest['label'] - 1

        # Split dataset
        np.random.seed(42)
        indices = np.random.permutation(len(self.manifest))
        split_idx = int(len(indices) * train_fraction)

        if split == 'train':
            selected_indices = indices[:split_idx]
        elif split == 'val':
            selected_indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.manifest = self.manifest.iloc[selected_indices].reset_index(drop=True)

        # Preload option
        self.preload_into_ram = preload_into_ram
        self.cache = {}

        if preload_into_ram:
            self._preload_data()

    def _preload_data(self):
        """Load all chunks into RAM."""
        print(f"Preloading dataset into RAM...")

        unique_chunks = self.manifest['chunk_file'].unique()
        for chunk_file in unique_chunks:
            chunk_path = self.chunks_dir / chunk_file
            # Load fully into RAM (no memory mapping)
            self.cache[chunk_file] = np.load(chunk_path)

        print(f"Loaded {len(unique_chunks)} chunks into RAM")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]

        # Load sequence
        if self.preload_into_ram:
            chunk_data = self.cache[row['chunk_file']]
        else:
            chunk_path = self.chunks_dir / row['chunk_file']
            chunk_data = np.load(chunk_path, mmap_mode='r')

        # Copy the array to ensure it's writable (avoids PyTorch warning)
        sequence = chunk_data[row['chunk_row']].copy()  # Shape: (5, 256)
        label = row['label']  # 0 or 1

        # Convert to torch tensors
        sequence = torch.from_numpy(sequence).float()  # (5, 256)
        label = torch.tensor(label, dtype=torch.long)

        return sequence, label


def create_dataloaders_v10(
    dataset_dir: str,
    batch_size: int = 512,
    workers: int = 8,
    preload: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for V10 sequences.

    Args:
        dataset_dir: Path to V10 dataset
        batch_size: Batch size
        workers: Number of dataloader workers
        preload: Preload dataset into RAM

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TemporalSequenceDataset(
        dataset_dir,
        split='train',
        preload_into_ram=preload
    )

    val_dataset = TemporalSequenceDataset(
        dataset_dir,
        split='val',
        preload_into_ram=preload
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader
