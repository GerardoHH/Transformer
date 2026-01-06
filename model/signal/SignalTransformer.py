import tensorflow as tf
from model.custom.encoder.Encoder import Encoder
from model.custom.decoder.Decoder import Decoder


class SignalTransformer(tf.keras.Model):
    """
    Transformer model adapted for continuous signal prediction.
    
    Architecture:
        Input signals → Encoder (N layers) → Context
        Target signals → Decoder (N layers) + Context → Dense → Predictions
    
    Replaces discrete vocabulary embeddings with continuous signal projections,
    making it suitable for time series forecasting and signal prediction tasks.
    
    Args:
        num_layers: Number of encoder/decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        dff: Feed-forward hidden dimension
        input_features: Number of input signal features (e.g., 4 for [q1, q2, q1dot, q2dot])
        target_features: Number of target signal features (typically 1)
        output_steps: Number of output time steps (for multi-step prediction)
        dropout_rate: Dropout rate
    """

    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_features, target_features=1, output_steps=1, 
                 dropout_rate=0.1):
        super().__init__()
        
        self.input_features = input_features
        self.target_features = target_features
        self.output_steps = output_steps
        
        self.encoder = Encoder(
            num_layers=num_layers, 
            d_model=d_model,
            num_heads=num_heads, 
            dff=dff,
            input_features=input_features,
            dropout_rate=dropout_rate)

        self.decoder = Decoder(
            num_layers=num_layers, 
            d_model=d_model,
            num_heads=num_heads, 
            dff=dff,
            target_features=target_features,
            dropout_rate=dropout_rate)

        # Final projection to target feature dimension
        self.final_layer = tf.keras.layers.Dense(target_features)

    def call(self, inputs, training=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            inputs: Tuple of (encoder_input, decoder_input)
                - encoder_input: (batch, input_seq_len, input_features)
                - decoder_input: (batch, target_seq_len, target_features)
            training: Boolean for training mode
                
        Returns:
            predictions: (batch, target_seq_len, target_features)
        """
        context, x = inputs

        # Encode input sequence
        context = self.encoder(context, training=training)  # (batch, input_seq_len, d_model)

        # Decode with attention to encoder output
        x = self.decoder(x, context, training=training)  # (batch, target_seq_len, d_model)

        # Project to target dimension
        predictions = self.final_layer(x)  # (batch, target_seq_len, target_features)

        return predictions
    
    def predict_sequence(self, encoder_input, start_token=None, max_length=None):
        """
        Autoregressive prediction for inference.
        
        Generates output sequence one step at a time, using previous predictions
        as input to the decoder.
        
        Args:
            encoder_input: Input signal tensor (batch, input_seq_len, input_features)
            start_token: Starting value(s) for decoder. If None, uses zeros.
                        Shape: (batch, 1, target_features)
            max_length: Maximum output sequence length. Defaults to self.output_steps.
            
        Returns:
            output_sequence: Generated predictions (batch, max_length, target_features)
        """
        if max_length is None:
            max_length = self.output_steps
            
        batch_size = tf.shape(encoder_input)[0]
        
        # Encode input sequence once
        context = self.encoder(encoder_input, training=False)
        
        # Initialize with start token (zeros if not provided)
        if start_token is None:
            decoder_input = tf.zeros((batch_size, 1, self.target_features))
        else:
            decoder_input = start_token
        
        # Generate sequence autoregressively
        outputs = []
        for _ in range(max_length):
            # Decode current sequence
            decoder_output = self.decoder(decoder_input, context, training=False)
            
            # Get prediction for last position
            last_pred = self.final_layer(decoder_output[:, -1:, :])  # (batch, 1, target_features)
            outputs.append(last_pred)
            
            # Append prediction to decoder input for next iteration
            decoder_input = tf.concat([decoder_input, last_pred], axis=1)
        
        # Stack all predictions
        output_sequence = tf.concat(outputs, axis=1)  # (batch, max_length, target_features)
        
        return output_sequence
    
    def get_attention_weights(self):
        """Get the last attention weights from the decoder for visualization."""
        return self.decoder.last_attn_scores
    
    def get_config(self):
        return {
            'num_layers': len(self.encoder.enc_layers),
            'd_model': self.encoder.d_model,
            'num_heads': self.encoder.enc_layers[0].self_attention.mha.num_heads,
            'dff': self.encoder.enc_layers[0].ffn.seq.layers[0].units,
            'input_features': self.input_features,
            'target_features': self.target_features,
            'output_steps': self.output_steps,
        }
