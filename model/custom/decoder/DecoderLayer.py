import tensorflow as tf
from ..attention.CausalSelfAttention import CausalSelfAttention
from ..attention.CrossAttention import CrossAttention
from ..FeedForward import FeedForward


class DecoderLayer(tf.keras.layers.Layer):
    """Single decoder layer: CausalSelfAttention → CrossAttention → FeedForward."""
    
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape: (batch_size, seq_len, d_model)
        return x
