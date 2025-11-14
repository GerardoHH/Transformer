import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from Plot import PlotUtils as pltU
from model.Transformer import SimpleTransformer 
from Dataset.DatasetUtils import MathTranslationDataset

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


def build_and_train_simpleTrasformer():
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
            pltU.plot_training_history(history)
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

    