import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==================== MODELO SIMPLIFICADO ====================
class SimpleTransformer(keras.Model):
    """
    Versión simplificada del Transformer para el ejemplo.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 input_vocab_size, target_vocab_size, max_seq_length,
                 dropout_rate=0.1):
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.encoder_embedding = layers.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = layers.Embedding(target_vocab_size, d_model)
        
        # Positional encoding (simplificado)
        self.pos_encoding = self.positional_encoding(max_seq_length, d_model)
        
        # Encoder layers
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
            for _ in range(num_layers)
        ]
        self.encoder_ffn = [
            keras.Sequential([
                layers.Dense(d_ff, activation='relu'),
                layers.Dense(d_model)
            ]) for _ in range(num_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
            for _ in range(num_layers)
        ]
        self.decoder_cross_attention = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
            for _ in range(num_layers)
        ]
        self.decoder_ffn = [
            keras.Sequential([
                layers.Dense(d_ff, activation='relu'),
                layers.Dense(d_model)
            ]) for _ in range(num_layers)
        ]
        
        # Layer normalization
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate)
        
        # Output layer
        self.output_layer = layers.Dense(target_vocab_size)
    
    def positional_encoding(self, length, depth):
        """Crea codificación posicional."""
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate([
            np.sin(angle_rads[:, 0::2]),
            np.cos(angle_rads[:, 1::2])
        ], axis=-1)
        
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)
    
    def encode(self, x, training=False):
        """Encoder del transformer."""
        # Embedding + positional encoding
        x = self.encoder_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        
        # Encoder layers
        for i in range(len(self.encoder_layers)):
            # Self-attention
            attn_output = self.encoder_layers[i](x, x, training=training)
            x = self.layernorm(x + attn_output)
            
            # Feed-forward
            ffn_output = self.encoder_ffn[i](x)
            x = self.layernorm(x + ffn_output)
        
        return x
    
    def decode(self, x, encoder_output, training=False, look_ahead_mask=None):
        """Decoder del transformer."""
        # Embedding + positional encoding
        x = self.decoder_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.dropout(x, training=training)
        
        # Decoder layers
        for i in range(len(self.decoder_layers)):
            # Masked self-attention
            attn_output = self.decoder_layers[i](
                x, x, training=training, use_causal_mask=True
            )
            x = self.layernorm(x + attn_output)
            
            # Cross-attention
            attn_output = self.decoder_cross_attention[i](
                x, encoder_output, training=training
            )
            x = self.layernorm(x + attn_output)
            
            # Feed-forward
            ffn_output = self.decoder_ffn[i](x)
            x = self.layernorm(x + ffn_output)
        
        return x
    
    def call(self, inputs, training=False):
        """Forward pass del modelo."""
        # Desempaquetar inputs correctamente
        if isinstance(inputs, (list, tuple)):
            encoder_input, decoder_input = inputs
        else:
            # Si solo se pasa un tensor, asumir que es el encoder input
            # y usar un decoder input vacío (para inferencia)
            encoder_input = inputs
            batch_size = tf.shape(encoder_input)[0]
            decoder_input = tf.zeros((batch_size, 1), dtype=tf.int32)
        
        # Encode
        encoder_output = self.encode(encoder_input, training)
        
        # Decode
        decoder_output = self.decode(decoder_input, encoder_output, training)
        
        # Output projection
        output = self.output_layer(decoder_output)
        
        return output

