import tensorflow as tf
from ..FeedForward import FeedForward
from ..attention.GlobalSelfAttention import GlobalSelfAttention


class EncoderLayer(tf.keras.layers.Layer):
    """Single encoder layer: GlobalSelfAttention → FeedForward."""
    
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
