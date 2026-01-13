import tensorflow as tf
from .EncoderLayer import EncoderLayer
from ..embedding.SignalEmbedding import SignalEmbedding


class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder adapted for continuous signals.
    
    Uses SignalEmbedding (Dense projection) instead of discrete token embedding.
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward hidden dimension
        input_features: Number of input features (replaces vocab_size)
        dropout_rate: Dropout rate
    """
    
    def __init__(self, *, num_layers, d_model, num_heads, dff, 
                 input_features, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # SignalEmbedding for continuous signals instead of PositionalEmbedding
        self.signal_embedding = SignalEmbedding(
            input_dim=input_features, 
            d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_features)
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Embed continuous signals: (batch, seq_len, input_features) -> (batch, seq_len, d_model)
        x = self.signal_embedding(x)

        # Add dropout
        x = self.dropout(x, training=training)

        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape: (batch_size, seq_len, d_model)
