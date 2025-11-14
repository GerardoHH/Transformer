import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class PositionalEncoding(layers.Layer):
    """
    Codificacion posicional para dar informacion de posicion 
    a los tokens. Usa funciones sinusoidales de diferentes frecuencias.
    """
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Crear la matriz de codificacion posicional
        self.positional_encoding = self.create_positional_encoding()
    
    def create_positional_encoding(self):
        position = np.arange(self.sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((self.sequence_length, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pos_encoding, dtype=tf.float32)[np.newaxis, ...]
    
    def call(self, x):
        # Agregar codificacion posicional a los embeddings
        return x + self.positional_encoding[:, :tf.shape(x)[1], :]

class MultiHeadAttention(layers.Layer):
    """
    Implementacion de Multi-Head Attention.
    Permite al modelo atender a informacion de diferentes posiciones
    desde diferentes subespacios de representacion.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        
        # Matrices de pesos para Q, K, V y salida
        self.W_q = layers.Dense(d_model)
        self.W_k = layers.Dense(d_model)
        self.W_v = layers.Dense(d_model)
        self.W_o = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Divide la ultima dimension en (num_heads, d_k)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]
        
        # 1. Proyectar Q, K, V
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    
        V = self.W_v(value)  
        
        # 2. Dividir en mltiples heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # 3. Calcular attention scores
        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        
        # Aplicar mascara si existe
        if mask is not None:
            scores = tf.where(mask == 0, -1e9, scores)
        
        # 4. Aplicar softmax
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # 5. Aplicar attention a valores
        attention_output = tf.matmul(attention_weights, V)
        
        # 6. Concatenar heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, 
                                     (batch_size, -1, self.d_model))
        
        # 7. Proyeccion final
        output = self.W_o(concat_attention)
        
        return output, attention_weights


class FeedForwardNetwork(layers.Layer):
    """
    Red feed-forward de dos capas con activacion ReLU.
    """
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = layers.Dense(d_ff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


# ==================== ENCODER ====================

class EncoderLayer(layers.Layer):
    """
    Capa individual del Encoder con:
    - Multi-Head Attention
    - Feed Forward Network
    - Conexiones residuales y normalizacion
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        # Multi-Head Attention con conexin residual
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        
        # Feed Forward con conexin residual
        ffn_output = self.ffn(x, training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        
        return x


class TransformerEncoder(layers.Layer):
    """
    Stack completo del Encoder.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 vocab_size, max_seq_length, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding y codificacin posicional
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_length, d_model)
        self.dropout = layers.Dropout(dropout_rate)
        
        # Stack de capas encoder
        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) 
                          for _ in range(num_layers)]
    
    def call(self, x, training, mask=None):
        # Obtener longitud de secuencia
        seq_len = tf.shape(x)[1]
        
        # Embedding y escalar por sqrt(d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Agregar codificacin posicional
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Pasar por cada capa del encoder
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x


# ==================== DECODER ====================

class DecoderLayer(layers.Layer):
    """
    Capa individual del Decoder con:
    - Masked Multi-Head Self-Attention
    - Multi-Head Cross-Attention
    - Feed Forward Network
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)  # Self-attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)  # Cross-attention
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # Masked self-attention
        attn1, attn_weights_1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        x = self.layernorm1(x + attn1)
        
        # Cross-attention con output del encoder
        attn2, attn_weights_2 = self.mha2(x, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        x = self.layernorm2(x + attn2)
        
        # Feed forward
        ffn_output = self.ffn(x, training)
        ffn_output = self.dropout3(ffn_output, training=training)
        x = self.layernorm3(x + ffn_output)
        
        return x, attn_weights_1, attn_weights_2


class TransformerDecoder(layers.Layer):
    """
    Stack completo del Decoder.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 vocab_size, max_seq_length, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding y codificacin posicional
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_length, d_model)
        self.dropout = layers.Dropout(dropout_rate)
        
        # Stack de capas decoder
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) 
                          for _ in range(num_layers)]
    
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # Embedding y escalar
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Agregar codificacin posicional
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Pasar por cada capa del decoder
        for i in range(self.num_layers):
            x, attn1, attn2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_attn1'] = attn1
            attention_weights[f'decoder_layer{i+1}_attn2'] = attn2
        
        return x, attention_weights


# ==================== MODELO COMPLETO ====================

class Transformer(keras.Model):
    """
    Modelo Transformer completo para tareas seq2seq.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 input_vocab_size, target_vocab_size, 
                 max_input_length, max_target_length,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder = TransformerEncoder(
            num_layers, d_model, num_heads, d_ff,
            input_vocab_size, max_input_length, dropout_rate
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            num_layers, d_model, num_heads, d_ff,
            target_vocab_size, max_target_length, dropout_rate
        )
        
        # Capa final de proyeccin
        self.final_layer = layers.Dense(target_vocab_size)
    
    def create_masks(self, inp, tar):
        """Crea las máscaras necesarias para el attention."""
        # Máscara de padding para el encoder
        enc_padding_mask = self.create_padding_mask(inp)
        
        # Máscara de padding para el decoder (usada en cross-attention)
        dec_padding_mask = self.create_padding_mask(inp)
        
        # Máscara look-ahead para el decoder (causal mask)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        
        # Máscara de padding para el target
        dec_target_padding_mask = self.create_padding_mask(tar)
        
        # Combinar máscaras look-ahead y padding
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def create_padding_mask(self, seq):
        """Crea máscara para ignorar tokens de padding (0)."""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        """Crea máscara triangular para prevenir attention a posiciones futuras."""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def call(self, inputs, training=False):
        inp, tar = inputs
        
        # Crear máscaras
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        
        # Encoder
        enc_output = self.encoder(inp, training, enc_padding_mask)
        
        # Decoder
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )
        
        # Proyeccin final
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights


# ==================== EJEMPLO DE USO ====================

def create_sample_model():
    """
    Crea un modelo Transformer de ejemplo con hiperparámetros típicos.
    """
    # Hiperparámetros
    num_layers = 4           # Número de capas encoder/decoder
    d_model = 128           # Dimensin del modelo
    num_heads = 8           # Número de attention heads
    d_ff = 512             # Dimensin de la red feed-forward
    dropout_rate = 0.1
    
    # Parámetros del vocabulario
    input_vocab_size = 8500
    target_vocab_size = 8000
    max_input_length = 50
    max_target_length = 50
    
    # Crear modelo
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        dropout_rate=dropout_rate
    )
    
    return transformer


# ==================== ENTRENAMIENTO ====================

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule personalizado como en el paper original.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """
    Funcin de pérdida con máscara para ignorar tokens de padding.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """
    Funcin de accuracy con máscara para ignorar tokens de padding.
    """
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


# ==================== EJEMPLO DE ENTRENAMIENTO ====================

def train_step_example():
    """
    Ejemplo de cmo entrenar el modelo.
    """
    # Crear modelo
    transformer = create_sample_model()
    
    # Configurar optimizador con schedule personalizado
    learning_rate = CustomSchedule(d_model=128)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # Compilar modelo
    transformer.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=[accuracy_function]
    )
    
    # Datos de ejemplo (normalmente vendrían de tu dataset)
    batch_size = 64
    max_length = 40
    
    # Datos ficticios para demostracin
    encoder_input = tf.random.uniform((batch_size, max_length), 
                                     minval=0, maxval=8500, dtype=tf.int32)
    decoder_input = tf.random.uniform((batch_size, max_length), 
                                     minval=0, maxval=8000, dtype=tf.int32)
    
    # Forward pass
    predictions, _ = transformer((encoder_input, decoder_input), training=True)
    
    print(f"Forma de entrada del encoder: {encoder_input.shape}")
    print(f"Forma de entrada del decoder: {decoder_input.shape}")
    print(f"Forma de las predicciones: {predictions.shape}")
    
    return transformer


# ==================== INFERENCIA ====================

def generate_text(transformer, start_token, end_token, encoder_input, max_length=50):
    """
    Genera texto usando el modelo Transformer entrenado.
    
    Args:
        transformer: Modelo Transformer entrenado
        start_token: Token de inicio de secuencia
        end_token: Token de fin de secuencia
        encoder_input: Input del encoder (texto fuente)
        max_length: Longitud máxima de generacin
    
    Returns:
        output: Secuencia generada
    """
    # Agregar dimensin batch si es necesario
    if len(encoder_input.shape) == 1:
        encoder_input = encoder_input[tf.newaxis, ...]
    
    # Token inicial para el decoder
    decoder_input = [start_token]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(max_length):
        # Obtener predicciones
        predictions, _ = transformer((encoder_input, output), training=False)
        
        # Obtener el token predicho (último token)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # Si es el token de fin, terminar
        if predicted_id == end_token:
            break
        
        # Concatenar el token predicho al output
        output = tf.concat([output, predicted_id], axis=-1)
    
    return output


# Crear y mostrar el modelo
print("Creando modelo Transformer...")
model = train_step_example()
print("\n¡Modelo creado exitosamente!")
    
# Mostrar resumen del modelo
print("\nResumen del modelo:")
print(f"Número de parámetros entrenables: {model.count_params():,}")

####################################################################
####################################################################

# ==================== IMPORTS ====================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import random

# Verificar versión de TensorFlow
print(f"TensorFlow versión: {tf.__version__}")
print(f"Keras versión: {keras.__version__}")

# ==================== DATASET: TRADUCCIÓN DE OPERACIONES MATEMÁTICAS ====================

class MathTranslationDataset:
    """
    Dataset simple que traduce operaciones matemáticas a su resultado en palabras.
    Ejemplo: "5 + 3" -> "eight"
    """
    def __init__(self, num_samples=10000, max_number=50):
        self.num_samples = num_samples
        self.max_number = max_number
        
        # Vocabularios
        self.input_tokens = ['<PAD>', '<START>', '<END>', '+', '-', '*', ' '] + \
                           [str(i) for i in range(100)]
        self.output_tokens = ['<PAD>', '<START>', '<END>'] + \
                           ['zero', 'one', 'two', 'three', 'four', 'five', 
                            'six', 'seven', 'eight', 'nine', 'ten',
                            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
                            'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'] + \
                           [f'twenty-{i}' for i in ['one', 'two', 'three', 'four', 'five', 
                                                    'six', 'seven', 'eight', 'nine']] + \
                           ['thirty'] + [f'thirty-{i}' for i in ['one', 'two', 'three', 'four', 
                                                                 'five', 'six', 'seven', 'eight', 'nine']] + \
                           ['forty'] + [f'forty-{i}' for i in ['one', 'two', 'three', 'four', 
                                                               'five', 'six', 'seven', 'eight', 'nine']] + \
                           ['fifty'] + [f'fifty-{i}' for i in ['one', 'two', 'three', 'four', 
                                                               'five', 'six', 'seven', 'eight', 'nine']] + \
                           ['sixty'] + [f'sixty-{i}' for i in ['one', 'two', 'three', 'four', 
                                                               'five', 'six', 'seven', 'eight', 'nine']] + \
                           ['seventy'] + [f'seventy-{i}' for i in ['one', 'two', 'three', 'four', 
                                                                   'five', 'six', 'seven', 'eight', 'nine']] + \
                           ['eighty'] + [f'eighty-{i}' for i in ['one', 'two', 'three', 'four', 
                                                                 'five', 'six', 'seven', 'eight', 'nine']] + \
                           ['ninety'] + [f'ninety-{i}' for i in ['one', 'two', 'three', 'four', 
                                                                 'five', 'six', 'seven', 'eight', 'nine']] + \
                           ['one-hundred', 'negative']
        
        # Crear mapeos token -> id
        self.input_token_to_id = {token: idx for idx, token in enumerate(self.input_tokens)}
        self.input_id_to_token = {idx: token for idx, token in enumerate(self.input_tokens)}
        
        self.output_token_to_id = {token: idx for idx, token in enumerate(self.output_tokens)}
        self.output_id_to_token = {idx: token for idx, token in enumerate(self.output_tokens)}
        
        # Tamaños de vocabulario
        self.input_vocab_size = len(self.input_tokens)
        self.output_vocab_size = len(self.output_tokens)
        
        # Tokens especiales
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        
    def number_to_words(self, n):
        """Convierte un número a palabras en inglés."""
        if n < 0:
            return f"negative {self.number_to_words(-n)}"
        
        words = ['zero', 'one', 'two', 'three', 'four', 'five', 
                'six', 'seven', 'eight', 'nine', 'ten',
                'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
                'sixteen', 'seventeen', 'eighteen', 'nineteen']
        
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 
                'sixty', 'seventy', 'eighty', 'ninety']
        
        if n < 20:
            return words[n]
        elif n < 100:
            ten = n // 10
            unit = n % 10
            if unit == 0:
                return tens[ten]
            else:
                return f"{tens[ten]}-{words[unit]}"
        elif n == 100:
            return "one-hundred"
        else:
            return "one-hundred"  # Simplificación para el ejemplo
    
    def generate_samples(self):
        """Genera pares de entrada/salida."""
        inputs = []
        outputs = []
        
        for _ in range(self.num_samples):
            # Elegir operación aleatoria
            op = random.choice(['+', '-', '*'])
            
            if op == '*':
                # Para multiplicación, usar números más pequeños
                a = random.randint(0, 10)
                b = random.randint(0, 10)
            else:
                a = random.randint(0, self.max_number)
                b = random.randint(0, self.max_number)
            
            # Crear expresión
            expr = f"{a} {op} {b}"
            
            # Calcular resultado
            if op == '+':
                result = a + b
            elif op == '-':
                result = a - b
            else:  # '*'
                result = a * b
            
            # Limitar resultado al rango manejable
            if -100 <= result <= 100:
                inputs.append(expr)
                outputs.append(self.number_to_words(result))
        
        return inputs, outputs
    
    def tokenize_input(self, text):
        """Tokeniza texto de entrada."""
        tokens = []
        for char in text:
            if char in self.input_token_to_id:
                tokens.append(self.input_token_to_id[char])
        return [self.start_token] + tokens + [self.end_token]
    
    def tokenize_output(self, text):
        """Tokeniza texto de salida."""
        # Para simplificar, tratamos cada palabra completa como un token
        if text in self.output_token_to_id:
            tokens = [self.output_token_to_id[text]]
        else:
            # Manejar números compuestos
            tokens = []
            parts = text.split()
            for part in parts:
                if part in self.output_token_to_id:
                    tokens.append(self.output_token_to_id[part])
        return [self.start_token] + tokens + [self.end_token]
    
    def prepare_dataset(self, batch_size=64, max_length=15):
        """
        Prepara el dataset para entrenamiento con Teacher Forcing.
        
        Teacher Forcing: Durante el entrenamiento, el decoder recibe la secuencia
        target correcta desplazada, no sus propias predicciones anteriores.
        """
        # Generar samples
        inputs, outputs = self.generate_samples()
        
        # Tokenizar
        input_ids = [self.tokenize_input(inp) for inp in inputs]
        output_ids = [self.tokenize_output(out) for out in outputs]
        
        # Padding para encoder inputs
        encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            input_ids, maxlen=max_length, padding='post', value=self.pad_token
        )
        
        # Padding para decoder outputs completos
        decoder_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            output_ids, maxlen=max_length, padding='post', value=self.pad_token
        )
        
        # Para Teacher Forcing:
        # - decoder_inputs: <START> token1 token2 ... (sin el último token)
        # - decoder_targets: token1 token2 ... <END> (sin el primer token START)
        # Esto permite que el modelo aprenda a predecir el siguiente token
        # basándose en los tokens anteriores correctos
        
        # Agregar START token al inicio para decoder inputs
        decoder_inputs = tf.concat([
            tf.ones((decoder_outputs.shape[0], 1), dtype=tf.int32) * self.start_token,
            decoder_outputs[:, :-1]
        ], axis=1)
        
        # Los targets son los outputs originales (ya incluyen END token)
        decoder_targets = decoder_outputs
        
        # Dividir en train/val
        split_idx = int(0.9 * len(encoder_inputs))
        
        train_encoder = encoder_inputs[:split_idx]
        train_decoder_in = decoder_inputs[:split_idx]
        train_targets = decoder_targets[:split_idx]
        
        val_encoder = encoder_inputs[split_idx:]
        val_decoder_in = decoder_inputs[split_idx:]
        val_targets = decoder_targets[split_idx:]
        
        # Crear datasets de TensorFlow
        # Formato: ((encoder_input, decoder_input), targets)
        train_dataset = tf.data.Dataset.from_tensor_slices((
            (train_encoder, train_decoder_in),
            train_targets
        ))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            (val_encoder, val_decoder_in),
            val_targets
        ))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, (inputs[:10], outputs[:10])
    
    def decode_predictions(self, predictions):
        """Decodifica predicciones a texto."""
        decoded = []
        for pred in predictions:
            tokens = []
            for token_id in pred:
                # Convertir a int si es necesario
                if hasattr(token_id, 'numpy'):
                    token_id = int(token_id.numpy())
                else:
                    token_id = int(token_id)
                    
                if token_id == self.end_token:
                    break
                if token_id not in [self.pad_token, self.start_token]:
                    if token_id in self.output_id_to_token:
                        tokens.append(self.output_id_to_token[token_id])
            decoded.append(' '.join(tokens) if tokens else '<empty>')
        return decoded


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


# ==================== FUNCIONES DE ENTRENAMIENTO ====================

def masked_loss(y_true, y_pred):
    """Función de pérdida con máscara para padding."""
    # y_true shape: (batch_size, seq_len)
    # y_pred shape: (batch_size, seq_len, vocab_size)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    # Crear máscara para ignorar padding
    mask = tf.cast(y_true != 0, dtype=loss.dtype)
    loss *= mask
    
    # Evitar división por cero
    mask_sum = tf.reduce_sum(mask)
    mask_sum = tf.maximum(mask_sum, 1.0)
    
    return tf.reduce_sum(loss) / mask_sum

def masked_accuracy(y_true, y_pred):
    """Función de accuracy con máscara para padding."""
    # y_true shape: (batch_size, seq_len)
    # y_pred shape: (batch_size, seq_len, vocab_size)
    
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, dtype=y_true.dtype)
    
    match = tf.cast(y_true == y_pred, dtype=tf.float32)
    mask = tf.cast(y_true != 0, dtype=tf.float32)
    
    # Evitar división por cero
    mask_sum = tf.reduce_sum(mask)
    mask_sum = tf.maximum(mask_sum, 1.0)
    
    return tf.reduce_sum(match * mask) / mask_sum


# ==================== ENTRENAMIENTO Y EVALUACIÓN ====================

def train_transformer(epochs=30, batch_size=64):
    """
    Entrena el modelo Transformer con el dataset de matemáticas.
    """
    print("=" * 50)
    print("PREPARANDO DATASET")
    print("=" * 50)
    
    # Crear dataset
    dataset = MathTranslationDataset(num_samples=5000, max_number=50)
    train_data, val_data, examples = dataset.prepare_dataset(batch_size=batch_size)
    
    print(f"Tamaño del vocabulario de entrada: {dataset.input_vocab_size}")
    print(f"Tamaño del vocabulario de salida: {dataset.output_vocab_size}")
    print(f"Ejemplos del dataset:")
    for inp, out in zip(examples[0][:5], examples[1][:5]):
        print(f"  {inp:15s} -> {out}")
    
    print("\n" + "=" * 50)
    print("CREANDO MODELO")
    print("=" * 50)
    
    # Crear modelo
    model = SimpleTransformer(
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=128,
        input_vocab_size=dataset.input_vocab_size,
        target_vocab_size=dataset.output_vocab_size,
        max_seq_length=20,
        dropout_rate=0.1
    )
    
    # Construir el modelo con un forward pass dummy
    dummy_encoder_input = tf.zeros((1, 15), dtype=tf.int32)
    dummy_decoder_input = tf.zeros((1, 15), dtype=tf.int32)
    _ = model([dummy_encoder_input, dummy_decoder_input], training=False)
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    
    print(f"Modelo creado con {model.count_params():,} parámetros")
    
    print("\n" + "=" * 50)
    print("ENTRENANDO MODELO")
    print("=" * 50)
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Entrenar
    history = model.fit( train_data, validation_data=val_data,
        epochs=epochs, callbacks=callbacks, verbose=2
    )
    
    return model, dataset, history


def evaluate_model(model, dataset, num_examples=10):
    """
    Evalúa el modelo con ejemplos de prueba.
    """
    print("\n" + "=" * 50)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 50)
    
    # Generar nuevos ejemplos de prueba
    test_inputs, test_outputs = dataset.generate_samples()
    test_inputs = test_inputs[:num_examples]
    test_outputs = test_outputs[:num_examples]
    
    # Tokenizar entradas
    input_ids = [dataset.tokenize_input(inp) for inp in test_inputs]
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(
        input_ids, maxlen=15, padding='post', value=0
    )
    
    # Generar predicciones con inferencia autoregresiva
    predictions = []
    
    for i in range(len(input_ids)):
        # Preparar encoder input
        encoder_input = tf.expand_dims(input_ids[i], 0)
        
        # Comenzar con token START
        decoder_input_list = [dataset.start_token]
        
        # Generar tokens uno por uno (inferencia autoregresiva)
        for step in range(10):
            # Convertir lista a tensor y agregar dimensión batch
            decoder_input = tf.expand_dims(
                tf.constant(decoder_input_list, dtype=tf.int32), 0
            )
            
            # Obtener predicción del modelo
            pred = model([encoder_input, decoder_input], training=False)
            
            # Obtener el token con mayor probabilidad del último timestep
            # pred shape: (1, seq_len, vocab_size)
            last_token_logits = pred[0, -1, :]  # (vocab_size,)
            next_token = tf.argmax(last_token_logits).numpy()
            
            # Si es END token, terminar
            if next_token == dataset.end_token:
                decoder_input_list.append(int(next_token))
                break
            
            # Agregar token predicho a la lista
            decoder_input_list.append(int(next_token))
        
        # Guardar la secuencia generada
        predictions.append(decoder_input_list)
    
    # Decodificar predicciones
    decoded_preds = dataset.decode_predictions(predictions)
    
    # Mostrar resultados
    print("\nResultados de prueba:")
    print("-" * 50)
    correct = 0
    for i in range(len(test_inputs)):
        is_correct = decoded_preds[i] == test_outputs[i]
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"{status} {test_inputs[i]:12s} = {test_outputs[i]:15s} | Predicción: {decoded_preds[i]}")
    
    accuracy = correct / len(test_inputs) * 100
    print(f"\nPrecisión: {accuracy:.1f}% ({correct}/{len(test_inputs)})")
    
    return accuracy


def plot_training_history(history):
    """
    Grafica el historial de entrenamiento.
    """
    plt.figure(figsize=(12, 4))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['masked_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_masked_accuracy'], label='Val Accuracy')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# ==================== PROGRAMA PRINCIPAL ====================

def main():
    """
    Función principal para ejecutar el ejemplo completo.
    """
    print("\n" + "=" * 50)
    print("TRANSFORMER: TRADUCTOR DE OPERACIONES MATEMÁTICAS")
    print("=" * 50)
    print("\nEste ejemplo entrena un Transformer para traducir")
    print("operaciones matemáticas a su resultado en palabras.")
    print("Ejemplo: '5 + 3' -> 'eight'\n")
    
    # Configurar semilla para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Entrenar modelo
        model, dataset, history = train_transformer(epochs=100, batch_size=32)
        
        # Evaluar modelo
        print("\nIniciando evaluación del modelo...")
        accuracy = evaluate_model(model, dataset, num_examples=20)
        
        # Graficar historial
        try:
            plot_training_history(history)
        except Exception as e:
            print(f"No se pudo graficar el historial: {e}")
        
        # Prueba interactiva
        print("\n" + "=" * 50)
        print("PRUEBA INTERACTIVA")
        print("=" * 50)
        print("\nPuedes probar el modelo con tus propias operaciones:")
        print("(Ingresa 'salir' para terminar)")
        
        while True:
            user_input = input("\nIngresa una operación (ej: '7 + 5'): ")
            if user_input.lower() == 'salir':
                break
            
            try:
                # Tokenizar entrada
                input_ids = dataset.tokenize_input(user_input)
                input_ids = tf.keras.preprocessing.sequence.pad_sequences(
                    [input_ids], maxlen=15, padding='post', value=0
                )
                
                # Preparar encoder input
                encoder_input = tf.expand_dims(input_ids[0], 0)
                
                # Lista para acumular tokens del decoder
                decoder_input_list = [dataset.start_token]
                
                # Generar predicción token por token
                for _ in range(10):
                    # Convertir lista a tensor
                    decoder_input = tf.expand_dims(
                        tf.constant(decoder_input_list, dtype=tf.int32), 0
                    )
                    
                    # Obtener predicción
                    pred = model([encoder_input, decoder_input], training=False)
                    
                    # Obtener siguiente token
                    last_token_logits = pred[0, -1, :]
                    next_token = tf.argmax(last_token_logits).numpy()
                    
                    # Si es END token, terminar
                    if next_token == dataset.end_token:
                        decoder_input_list.append(int(next_token))
                        break
                    
                    # Agregar token a la lista
                    decoder_input_list.append(int(next_token))
                
                # Decodificar resultado
                result = dataset.decode_predictions([decoder_input_list])[0]
                
                # Calcular respuesta correcta
                try:
                    correct_result = eval(user_input)
                    correct_words = dataset.number_to_words(correct_result)
                    print(f"Resultado: {result}")
                    print(f"Correcto: {correct_words} ({correct_result})")
                except:
                    print(f"Resultado: {result}")
            
            except Exception as e:
                print(f"Error: {e}")
                print("Intenta con una operación simple como '5 + 3'")
    
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        print("\nDetalles del error:")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("DEPURACIÓN")
        print("=" * 50)
        print("Verificando componentes individuales...")
        
        try:
            # Probar dataset
            print("1. Probando dataset...")
            test_dataset = MathTranslationDataset(num_samples=100)
            test_train, test_val, _ = test_dataset.prepare_dataset(batch_size=16)
            print("   ✓ Dataset funciona correctamente")
            
            # Probar modelo
            print("2. Probando modelo...")
            test_model = SimpleTransformer(
                num_layers=1, d_model=32, num_heads=2, d_ff=64,
                input_vocab_size=test_dataset.input_vocab_size,
                target_vocab_size=test_dataset.output_vocab_size,
                max_seq_length=20, dropout_rate=0.1
            )
            
            # Probar forward pass
            for batch in test_train.take(1):
                inputs, targets = batch
                output = test_model(inputs, training=False)
                print(f"   ✓ Forward pass funciona - Output shape: {output.shape}")
            
        except Exception as debug_error:
            print(f"   ✗ Error en componentes: {debug_error}")
    
    print("\n¡Gracias por probar el Transformer!")


if __name__ == "__main__":
    main()

###  PRueba GIT!!!
### From Lap
