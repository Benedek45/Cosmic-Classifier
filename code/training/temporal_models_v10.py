"""
Temporal Models V10 - LSTM and Transformer architectures

Models for temporal sequence classification on V10 data.
Input shape: (batch, 5, 256) - 5 windows per sequence
Output: 2 classes (0=negative, 1=transit)
"""

import torch
import torch.nn as nn
import math


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal sequence classification.

    Architecture:
    - Bidirectional LSTM (2 layers)
    - Dropout for regularization
    - Fully connected classifier
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        # Bidirectional doubles the hidden dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - e.g., (batch, 5, 256)

        Returns:
            logits: (batch, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state from both directions
        # h_n: (num_layers * 2, batch, hidden_dim)
        h_forward = h_n[-2, :, :]  # Last layer, forward direction
        h_backward = h_n[-1, :, :]  # Last layer, backward direction

        # Concatenate bidirectional outputs
        h_concat = torch.cat([h_forward, h_backward], dim=1)

        # Dropout
        h_concat = self.dropout(h_concat)

        # Classifier
        logits = self.fc(h_concat)

        return logits


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal sequence classification.

    Architecture:
    - Input projection
    - Positional encoding
    - Transformer encoder layers
    - Classification head
    """

    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()

        # Project input to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - e.g., (batch, 5, 256)

        Returns:
            logits: (batch, num_classes)
        """
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits


def get_temporal_model(model_name: str) -> nn.Module:
    """
    Factory function to create temporal models.

    Args:
        model_name: Name of the model ('temporal_lstm' or 'temporal_transformer')

    Returns:
        Initialized model
    """
    models = {
        'temporal_lstm': TemporalLSTM,
        'temporal_transformer': TemporalTransformer
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models.keys())}"
        )

    model = models[model_name]()

    return model


# Aliases for inference script compatibility
TemporalCNNLSTM = TemporalLSTM
TemporalCNNTransformer = TemporalTransformer


if __name__ == "__main__":
    # Test models
    print("Testing temporal models...")

    # Create dummy input
    batch_size = 8
    seq_len = 5
    input_dim = 256
    x = torch.randn(batch_size, seq_len, input_dim)

    # Test LSTM
    print("\nTesting TemporalLSTM...")
    lstm_model = get_temporal_model('temporal_lstm')
    lstm_out = lstm_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {lstm_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    # Test Transformer
    print("\nTesting TemporalTransformer...")
    transformer_model = get_temporal_model('temporal_transformer')
    transformer_out = transformer_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {transformer_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")

    print("\n✅ All models working correctly!")
