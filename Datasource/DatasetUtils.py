import random
import tensorflow as tf

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
