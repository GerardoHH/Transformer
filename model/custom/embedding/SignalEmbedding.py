import tensorflow as tf
import numpy as np


def positional_encoding(length, depth):
    """Sinusoidal positional encoding from 'Attention Is All You Need'."""
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)         # (1, depth)
    angle_rads = positions * angle_rates        # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class SignalEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer for continuous signals.
    
    Unlike PositionalEmbedding which uses discrete token embeddings,
    this uses a Dense projection for continuous signal values.
    
    Args:
        input_dim: Number of input features (e.g., 4 for [q1, q2, q1dot, q2dot])
        d_model: Model dimension for projection
        max_length: Maximum sequence length for positional encoding
    """
    
    def __init__(self, input_dim, d_model, max_length=2048):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Linear projection from input features to d_model
        self.projection = tf.keras.layers.Dense(d_model)
        
        # Precomputed positional encoding
        self.pos_encoding = positional_encoding(length=max_length, depth=d_model)

    def call(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim) - continuous signal values
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        length = tf.shape(x)[1]
        
        # Project input features to d_model dimensions
        x = self.projection(x)  # (batch_size, seq_len, d_model)
        
        # Scale by sqrt(d_model) as in original Transformer
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Add positional encoding
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'd_model': self.d_model,
        })
        return config
