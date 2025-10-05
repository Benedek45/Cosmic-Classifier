"""
Convert PyTorch Lightning .ckpt checkpoint to standalone .pt file for easier inference.
This removes all Lightning-specific metadata and keeps only the model weights.
"""
import torch
import sys
from pathlib import Path
import argparse
from dataclasses import dataclass


# Required for unpickling checkpoints from V3 trainer
@dataclass
class MultiArchConfig:
    """Config class from V3 trainer - needed to unpickle checkpoints"""
    data_dir: str = "extracted_windows_safe"
    positive_windows: str = "positive_windows_normalized.npy"
    negative_windows: str = "negative_windows_normalized.npy"
    positive_metadata: str = "positive_metadata.csv"
    negative_metadata: str = "negative_metadata.csv"
    batch_size: int = 1024
    num_epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 15
    samples_per_epoch: int = 4_000_000
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    window_size: int = 256
    dropout: float = 0.3
    weight_decay: float = 0.01
    num_workers: int = 16
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    precision: str = '16-mixed'
    random_seed: int = 42
    checkpoint_dir: str = "checkpoints_multi_arch"
    results_dir: str = "multi_arch_results"
    architectures: list = None

    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ['cnn', 'resnet', 'attention']


def convert_checkpoint(ckpt_path, output_path=None, architecture='cnn'):
    """
    Convert Lightning checkpoint to standalone .pt file

    Args:
        ckpt_path: Path to .ckpt file
        output_path: Optional output path (default: same name with .pt extension)
        architecture: Model architecture ('cnn', 'resnet', 'attention')
    """
    print(f"\n{'='*80}")
    print("CHECKPOINT TO .PT CONVERTER")
    print(f"{'='*80}\n")

    # Convert to Path object and make relative paths relative to current working directory
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path('.') / ckpt_path

    if not ckpt_path.exists():
        print(f" Error: Checkpoint not found: {ckpt_path}")
        return False

    # Load checkpoint
    print(f" Loading: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint, dict):
        print(f" Error: Invalid checkpoint format")
        return False

    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f" Found state_dict with {len(state_dict)} keys")
    else:
        print(f" Error: No state_dict in checkpoint")
        return False

    # Clean up keys (remove Lightning prefixes)
    cleaned_state_dict = {}
    removed_prefixes = set()

    for key, value in state_dict.items():
        new_key = key

        # Remove 'model.' prefix
        if new_key.startswith('model.'):
            new_key = new_key[6:]
            removed_prefixes.add('model.')

        # Remove 'model_arch.' prefix
        if new_key.startswith('model_arch.'):
            new_key = new_key[11:]
            removed_prefixes.add('model_arch.')

        # Remove '_orig_mod.' prefix (from torch.compile)
        if new_key.startswith('_orig_mod.'):
            new_key = new_key[10:]
            removed_prefixes.add('_orig_mod.')

        # Skip non-model keys (metrics, etc.)
        if any(x in new_key for x in ['criterion', 'train_', 'val_', 'test_']):
            continue

        cleaned_state_dict[new_key] = value

    if removed_prefixes:
        print(f" Removed prefixes: {', '.join(sorted(removed_prefixes))}")

    print(f" Cleaned state_dict: {len(cleaned_state_dict)} keys")

    # Show sample keys
    sample_keys = list(cleaned_state_dict.keys())[:5]
    print(f"\n Sample keys:")
    for key in sample_keys:
        print(f"   - {key}")
    if len(cleaned_state_dict) > 5:
        print(f"   ... and {len(cleaned_state_dict) - 5} more")

    # Determine model parameters from state_dict
    # Infer dropout from architecture
    model_config = {
        'window_size': 256,  # Fixed for this model
        'num_classes': 1,    # Binary classification
    }

    # Architecture-specific defaults
    if architecture == 'cnn':
        model_config['dropout'] = 0.3
    elif architecture == 'resnet':
        model_config['dropout'] = 0.3
        model_config['num_blocks'] = 3
    elif architecture == 'attention':
        model_config['dropout'] = 0.3
        model_config['num_heads'] = 4
        model_config['num_layers'] = 2

    # Create output dictionary with ALL necessary info
    output_dict = {
        'state_dict': cleaned_state_dict,
        'architecture': architecture,
        'model_config': model_config,
        'inference_config': {
            'threshold': 0.5,
            'bls_params': {
                'min_period': 1.0,
                'max_period': 30.0,
                'n_periods': 5000,
                'min_duration': 0.01,
                'max_duration': 0.2,
                'n_durations': 15,
                'cadence_minutes': 30.0
            },
            'window_size': 256,
            'sliding_stride': 128
        },
        'metadata': {
            'original_checkpoint': str(ckpt_path),
            'num_parameters': sum(p.numel() for p in cleaned_state_dict.values()),
            'converter_version': '2.0'
        }
    }

    # Extract additional metadata if available
    if 'hyper_parameters' in checkpoint:
        hyper = checkpoint['hyper_parameters']

        # Extract useful fields from hyperparameters (but don't save the whole object)
        if hasattr(hyper, 'dropout'):
            model_config['dropout'] = hyper.dropout
        if hasattr(hyper, 'window_size'):
            model_config['window_size'] = hyper.window_size

        # Store only serializable hyperparameters (no class instances)
        serializable_hyper = {}
        if hasattr(hyper, '__dict__'):
            for key, value in hyper.__dict__.items():
                # Only save primitive types
                if isinstance(value, (int, float, str, bool, type(None))):
                    serializable_hyper[key] = value
                elif isinstance(value, list) and all(isinstance(v, (int, float, str, bool)) for v in value):
                    serializable_hyper[key] = value

            if serializable_hyper:
                output_dict['metadata']['hyper_parameters'] = serializable_hyper

    if 'epoch' in checkpoint:
        output_dict['metadata']['epoch'] = checkpoint['epoch']

    # Determine output path
    if output_path is None:
        output_path = ckpt_path.parent / f"{ckpt_path.stem}.pt"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = Path('.') / output_path

    # Save
    print(f"\n Saving to: {output_path}")
    torch.save(output_dict, output_path)

    # Verify
    print(f"\n Verifying...")
    test_load = torch.load(output_path, map_location='cpu', weights_only=False)
    assert 'state_dict' in test_load
    assert len(test_load['state_dict']) == len(cleaned_state_dict)

    # Show file sizes
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    pt_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f" Conversion successful!")
    print(f"\n File sizes:")
    print(f"   Original .ckpt: {ckpt_size_mb:.2f} MB")
    print(f"   Converted .pt:  {pt_size_mb:.2f} MB")
    print(f"   Space saved:    {ckpt_size_mb - pt_size_mb:.2f} MB ({(1 - pt_size_mb/ckpt_size_mb)*100:.1f}%)")

    print(f"\n Model Configuration:")
    print(f"   Architecture:   {architecture}")
    print(f"   Window size:    {model_config['window_size']}")
    print(f"   Dropout:        {model_config['dropout']}")
    print(f"   Parameters:     {output_dict['metadata']['num_parameters']:,}")

    print(f"\n  Inference Settings (embedded):")
    print(f"   Threshold:      {output_dict['inference_config']['threshold']}")
    print(f"   BLS periods:    {output_dict['inference_config']['bls_params']['min_period']}-{output_dict['inference_config']['bls_params']['max_period']} days")
    print(f"   Sliding stride: {output_dict['inference_config']['sliding_stride']}")

    print(f"\n{'='*80}")
    print(f" DONE! This .pt file is self-contained and ready for inference")
    print(f"   No need to specify architecture - it's auto-detected!")
    print(f"{'='*80}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description='Convert Lightning checkpoint to .pt')
    parser.add_argument('checkpoint', help='Path to .ckpt file')
    parser.add_argument('-o', '--output', help='Output .pt file path (optional)')
    parser.add_argument('-a', '--architecture', default='cnn',
                       choices=['cnn', 'resnet', 'attention'],
                       help='Model architecture (default: cnn)')

    args = parser.parse_args()

    success = convert_checkpoint(args.checkpoint, args.output, args.architecture)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
