import tensorflow as tf
from ..embedding.SignalEmbedding import SignalEmbedding
from .DecoderLayer import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder adapted for continuous signals.
    
    Uses SignalEmbedding (Dense projection) instead of discrete token embedding.
    For signal prediction, the decoder receives target signal values (teacher forcing
    during training) and generates predictions autoregressively during inference.
    
    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward hidden dimension
        target_features: Number of target features (typically 1 for single signal prediction)
        dropout_rate: Dropout rate
    """
    
    def __init__(self, *, num_layers, d_model, num_heads, dff, 
                 target_features, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # SignalEmbedding for continuous target signals
        self.signal_embedding = SignalEmbedding(
            input_dim=target_features, 
            d_model=d_model)
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        self.dec_layers = [
            DecoderLayer(d_model=d_model, 
                         num_heads=num_heads,
                         dff=dff, 
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context, training=None):
        """
        Args:
            x: Target signal tensor of shape (batch_size, target_seq_len, target_features)
            context: Encoder output of shape (batch_size, context_seq_len, d_model)
            
        Returns:
            Tensor of shape (batch_size, target_seq_len, d_model)
        """
        # Embed target signals: (batch, target_seq_len, target_features) -> (batch, target_seq_len, d_model)
        x = self.signal_embedding(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # Shape: (batch_size, target_seq_len, d_model)
        return x
