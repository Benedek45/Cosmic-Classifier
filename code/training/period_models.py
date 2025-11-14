"""
Neural Network Models for Period Detection

Models that take full lightcurve + Model 1 timeline and predict period directly.

Architectures:
1. RobustPeriodNet: Dual-input (flux + timeline) with adaptive downsampling
2. FourierPeriodNet: Uses FFT features for periodicity detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AdaptiveDownsampler(nn.Module):
    """Smart downsampling for variable-length sequences using interpolation."""

    def __init__(self, target_length: int = 2048):
        super().__init__()
        self.target_length = target_length

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, max_len, channels) - padded sequences
            lengths: (batch,) - actual length of each sequence

        Returns:
            x_resampled: (batch, target_length, channels)
        """
        batch_size, max_len, channels = x.shape
        device = x.device

        # Resample each sequence to target length
        resampled = []

        for i in range(batch_size):
            actual_len = lengths[i].item()
            seq = x[i, :actual_len, :]  # (actual_len, channels)

            # Reshape for interpolation: (channels, actual_len)
            seq_reshaped = seq.transpose(0, 1).unsqueeze(0)  # (1, channels, actual_len)

            # Interpolate to target length
            seq_resampled = F.interpolate(
                seq_reshaped,
                size=self.target_length,
                mode='linear',
                align_corners=False
            )

            # Reshape back: (target_length, channels)
            seq_final = seq_resampled.squeeze(0).transpose(0, 1)

            resampled.append(seq_final)

        return torch.stack(resampled)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class RobustPeriodNet(nn.Module):
    """
    Robust period detection network with variable-length handling.

    Three input branches:
    1. Raw flux (detrended, normalized)
    2. Model 1 timeline (transit probabilities)
    3. Fourier features (frequency domain)

    Architecture:
    - Adaptive downsampling to fixed length
    - Dual CNN encoders for flux and timeline
    - Fourier feature encoder
    - Transformer for temporal dependencies
    - Period regression head with log-scale prediction
    """

    def __init__(
        self,
        target_length: int = 2048,
        hidden_dim: int = 256,
        num_transformer_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.target_length = target_length

        # Adaptive downsampler
        self.downsampler = AdaptiveDownsampler(target_length)

        # ============================================
        # Branch 1: Flux Encoder (1D CNN + Transformer)
        # ============================================
        self.flux_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=31, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(256, max_len=target_length, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.flux_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # ============================================
        # Branch 2: Timeline Encoder (Simpler, just CNN)
        # ============================================
        self.timeline_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
        )

        # ============================================
        # Branch 3: Fourier Features
        # ============================================
        self.fourier_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # ============================================
        # Fusion Layer
        # ============================================
        # Input: 256 (flux) + 128 (timeline) + 128 (fourier) = 512
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
        )

        # ============================================
        # Output Heads
        # ============================================
        # Period regression (log-scale for better range)
        self.period_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, flux, timeline, lengths):
        """
        Args:
            flux: (batch, max_len) - padded flux values
            timeline: (batch, max_len) - padded timeline values
            lengths: (batch,) - actual lengths

        Returns:
            period_log: (batch, 1) - predicted log(period)
            confidence: (batch, 1) - confidence score 0-1
        """
        # Downsample/upsample to target length
        flux_stacked = torch.stack([flux, timeline], dim=2)  # (batch, max_len, 2)
        flux_resampled = self.downsampler(flux_stacked, lengths)  # (batch, target_len, 2)

        flux_res = flux_resampled[:, :, 0]  # (batch, target_len)
        timeline_res = flux_resampled[:, :, 1]  # (batch, target_len)

        # ============================================
        # Branch 1: Flux encoding
        # ============================================
        # CNN
        flux_feat = self.flux_cnn(flux_res.unsqueeze(1))  # (batch, 256, target_len)
        flux_feat = flux_feat.transpose(1, 2)  # (batch, target_len, 256)

        # Positional encoding
        flux_feat = self.pos_encoder(flux_feat)

        # Transformer
        flux_feat = self.flux_transformer(flux_feat)  # (batch, target_len, 256)

        # Global pooling
        flux_feat = flux_feat.mean(dim=1)  # (batch, 256)

        # ============================================
        # Branch 2: Timeline encoding
        # ============================================
        timeline_feat = self.timeline_cnn(timeline_res.unsqueeze(1))  # (batch, 128, target_len)
        timeline_feat = timeline_feat.mean(dim=2)  # (batch, 128)

        # ============================================
        # Branch 3: Fourier features
        # ============================================
        # FFT on flux (differentiable)
        fft = torch.fft.rfft(flux_res, dim=1)
        fft_mag = torch.abs(fft)  # (batch, freq_bins)

        # Pad/truncate to fixed size
        if fft_mag.shape[1] < 512:
            fft_mag = F.pad(fft_mag, (0, 512 - fft_mag.shape[1]))
        else:
            fft_mag = fft_mag[:, :512]

        fourier_feat = self.fourier_encoder(fft_mag)  # (batch, 128)

        # ============================================
        # Fusion
        # ============================================
        combined = torch.cat([flux_feat, timeline_feat, fourier_feat], dim=1)
        fused = self.fusion(combined)  # (batch, 256)

        # ============================================
        # Outputs
        # ============================================
        # Period (log-scale, will apply exp() later)
        period_log = self.period_head(fused)  # (batch, 1)

        # Confidence
        confidence = self.confidence_head(fused)  # (batch, 1)

        return period_log, confidence


class FourierPeriodNet(nn.Module):
    """
    Simplified period detection using primarily Fourier features.

    Faster and more interpretable than RobustPeriodNet.
    Good for lightcurves with strong periodic signals.
    """

    def __init__(
        self,
        max_freqs: int = 1000,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()

        # Learnable Fourier feature extraction
        self.freq_encoder = nn.Sequential(
            nn.Linear(max_freqs, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Timeline CNN (lightweight)
        self.timeline_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 31, padding=15),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(32 * 128, 128),
            nn.ReLU(),
        )

        # Combine features
        self.combiner = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.period_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, flux, timeline):
        """
        Args:
            flux: (batch, length) - flux values (can be variable, will be padded internally)
            timeline: (batch, length) - timeline values

        Returns:
            period_log: (batch, 1) - predicted log(period)
            confidence: (batch, 1) - confidence score
        """
        # Compute Fourier features (differentiable)
        fft = torch.fft.rfft(flux, dim=1)
        fft_mag = torch.abs(fft)  # (batch, freqs)

        # Pad/truncate to 1000
        if fft_mag.shape[1] < 1000:
            fft_mag = F.pad(fft_mag, (0, 1000 - fft_mag.shape[1]))
        else:
            fft_mag = fft_mag[:, :1000]

        # Encode Fourier features
        freq_features = self.freq_encoder(fft_mag)  # (batch, 256)

        # Encode timeline
        timeline_features = self.timeline_encoder(timeline.unsqueeze(1))  # (batch, 128)

        # Combine
        combined = torch.cat([freq_features, timeline_features], dim=1)
        fused = self.combiner(combined)  # (batch, 128)

        # Outputs
        period_log = self.period_head(fused)
        confidence = self.confidence_head(fused)

        return period_log, confidence


if __name__ == "__main__":
    # Test models with variable-length input
    print("Testing RobustPeriodNet...")

    model = RobustPeriodNet(target_length=2048)

    # Simulate batch with different lengths
    batch_size = 4
    lengths = torch.tensor([500, 2000, 10000, 50000])
    max_len = lengths.max().item()

    # Padded inputs
    flux = torch.randn(batch_size, max_len)
    timeline = torch.randn(batch_size, max_len)

    # Forward
    period_log, confidence = model(flux, timeline, lengths)

    # Convert log period to actual period
    period = torch.exp(period_log)

    print(f"Input lengths: {lengths.tolist()}")
    print(f"Predicted log periods: {period_log.squeeze().tolist()}")
    print(f"Predicted periods: {period.squeeze().tolist()}")
    print(f"Confidence scores: {confidence.squeeze().tolist()}")
    print("\n✅ RobustPeriodNet works with variable lengths!")

    print("\nTesting FourierPeriodNet...")
    fourier_model = FourierPeriodNet()

    # Fixed length for simpler model
    flux_fixed = torch.randn(4, 2048)
    timeline_fixed = torch.randn(4, 2048)

    period_log2, conf2 = fourier_model(flux_fixed, timeline_fixed)
    print(f"Predicted periods: {torch.exp(period_log2).squeeze().tolist()}")
    print(f"Confidence: {conf2.squeeze().tolist()}")
    print("\n✅ FourierPeriodNet works!")
