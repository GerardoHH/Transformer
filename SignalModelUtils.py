import os
import sys
import json
import random
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime


from Datasource.SignalDataset import SignalDataset, create_signal_dataset
from model.signal.SignalTransformer import SignalTransformer
from model.utils.SignalTrainingUtils import (
    TransformerLRSchedule,
    CosineDecayWithWarmup,
    create_callbacks,
    plot_training_history,
    plot_predictions,
    plot_multistep_prediction,
    plot_full_signal_comparison,
    plot_multistep_prediction,
    SignalMetrics
)
def enviroment():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    print(f"Enviroment: ")
    print(f"\tPython {sys.version}")
    print(f"\tNumpy  {np.__version__}")
    print(f"\tTensor Flow Version: {tf.__version__}")
    print(f"\tKeras Version: {tf.keras.__version__}")
    
    print(f"\tEager execution: {tf.executing_eagerly()} ")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n\tGPU available")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\t   Memory growth habilitado")
    else:
        print("\n  No se detectó GPU, usando CPU")
    return len(gpus) > 0
        
    print("-----------------------------------------------------------------")

def reproducibility():
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    
"""
Script de entrenamiento para Signal Transformer
================================================

Ejemplo de uso:
    python train.py --input_window 100 --output_window 20 --epochs 50

Para ver todos los parámetros:
    python train.py --help
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Entrenar Signal Transformer')
    
    # Dataset
    parser.add_argument('--data_path', type=str, 
                        default='./data/true_dynamics.csv',
                        help='Ruta al archivo CSV')
    parser.add_argument('--input_features', type=str, nargs='+',
                        default=['q1 [rad]', 'q2 [rad]', 'q1dot [rad/s]', 'q2dot [rad/s]'],
                        help='Features de entrada')
    parser.add_argument('--target_feature', type=str, 
                        default='q1 [rad]',
                        help='Feature objetivo a predecir')
    parser.add_argument('--input_window', type=int, default=100,
                        help='Tamaño de ventana de entrada')
    parser.add_argument('--output_window', type=int, default=20,
                        help='Tamaño de ventana de salida (pasos a predecir)')
    parser.add_argument('--stride', type=int, default=10,
                        help='Paso entre ventanas')
    parser.add_argument('--scaler', type=str, default='standard',
                        choices=['standard', 'minmax'],
                        help='Tipo de escalador')
    
    # Modelo
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Número de capas encoder/decoder')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimensión del modelo')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Número de cabezas de atención')
    parser.add_argument('--dff', type=int, default=512,
                        help='Dimensión feed-forward')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Tasa de dropout')
    
    # Entrenamiento
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Tamaño de batch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate inicial')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Pasos de warmup')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'transformer', 'constant'],
                        help='Tipo de learning rate schedule')
    parser.add_argument('--patience', type=int, default=15,
                        help='Paciencia para early stopping')
    
    # Outputs
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directorio de salida')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Nombre del experimento')
    
    return parser.parse_args()


def main():
    args = parse_args()

    #Custom parms
    args.data_path = os.getcwd()+os.sep+"Dataset"+os.sep+"signal" + os.sep + "true_dynamics.csv"
    args.epochs = 25

    args.num_layers = 2
    args.d_model =64
    args.num_heads = 2
    args.dff = 64
    
    print("\n" + "="*60)
    print("\nHiperparameters")
    print(f"\tnum_layers: {args.num_layers}")
    print(f"\td_model: {args.d_model}")
    print(f"\tnum_heads: {args.num_heads}")
    print(f"\td_ff: {args.dff}")
    print(f"\tinput_features: {args.input_features}")
    print(f"\toutput_window: {args.output_window}")
    print(f"\tdropout: {args.dropout}")
    print("\n" + "="*60)
    
    base_output = os.path.join(os.getcwd(),'Outputs')
    
    
    checkpoint_dir = os.path.join(base_output, 'checkpoints')
    log_dir = os.path.join(base_output, 'logs')
    exp_dir = os.path.join(base_output, 'metrics')

    """
     filepath=args.data_path,
        input_features=args.input_features,
        target_feature=args.target_feature,
        input_window=args.input_window,
        output_window=args.output_window,
        stride=args.stride,
        batch_size=args.batch_size,
        scaler_type=args.scaler
    """

    #args.filepath= ""
    #python train.py --input_window 100 --output_window 20 --epochs 50

    #os.makedirs(exp_dir, exist_ok=True)
    #os.makedirs(checkpoint_dir, exist_ok=True)
    #os.makedirs(log_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SIGNAL TRANSFORMER - ENTRENAMIENTO")
    print("="*60)
        
    # =========================================================================
    # Dataset
    # =========================================================================
    print("\n" + "-"*60)
    print("CARGANDO DATASET")
    print("-"*60)
    
    dataset, train_ds, val_ds, test_ds = create_signal_dataset(
        filepath=args.data_path,
        input_features=args.input_features,
        target_feature=args.target_feature,
        input_window=args.input_window,
        output_window=args.output_window,
        stride=args.stride,
        batch_size=args.batch_size,
        scaler_type=args.scaler
    )
    
    # =========================================================================
    # Modelo
    # =========================================================================
    print("\n" + "-"*60)
    print("CREANDO MODELO")
    print("-"*60)
    
    model = SignalTransformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        input_features=len(args.input_features),
        output_steps=args.output_window,
        dropout_rate=args.dropout
    )
    
    # Build model para ver summary
    sample_enc = tf.zeros((1, args.input_window, len(args.input_features)))
    sample_dec = tf.zeros((1, args.output_window, 1))
    _ = model((sample_enc, sample_dec))
    
    model.summary()
    
    # =========================================================================
    # Learning Rate Schedule
    # =========================================================================
    if args.lr_schedule == 'transformer':
        lr_schedule = TransformerLRSchedule(args.d_model, args.warmup_steps)
    elif args.lr_schedule == 'cosine':
        # Estimar total de steps
        steps_per_epoch = len(list(train_ds))
        total_steps = steps_per_epoch * args.epochs
        lr_schedule = CosineDecayWithWarmup(
            initial_lr=args.lr,
            warmup_steps=args.warmup_steps,
            decay_steps=total_steps
        )
    else:
        lr_schedule = args.lr
    
    # =========================================================================
    # Compilar
    # =========================================================================
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print(f"\nOptimizer: Adam")
    print(f"LR Schedule: {args.lr_schedule}")
    print(f"Loss: MSE")
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    callbacks = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        patience=args.patience,
        model_name='signal_transformer'
    )
    
    # =========================================================================
    # Entrenamiento
    # =========================================================================
    print("\n" + "-"*60)
    print("INICIANDO ENTRENAMIENTO")
    print("-"*60)
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {args.patience}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    # =========================================================================
    # Guardar modelo final
    # =========================================================================
    model.save_weights(os.path.join(checkpoint_dir, 'signal_transformer_final.weights.h5'))
    print(f"\n Modelo guardado en {checkpoint_dir}")
    
    # =========================================================================
    # Evaluación
    # =========================================================================
    print("\n" + "-"*60)
    print("EVALUACIÓN EN TEST SET")
    print("-"*60)
    
    # Cargar mejores pesos
    model.load_weights(os.path.join(checkpoint_dir, 'signal_transformer_best.weights.h5'))
    
    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # Predicciones en test set
    print("\nGenerando predicciones...")
    all_preds = []
    all_targets = []
    
    for (enc_in, dec_in), target in test_ds:
        preds = model((enc_in, dec_in), training=False)
        all_preds.append(preds.numpy())
        all_targets.append(target.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Métricas detalladas
    metrics = SignalMetrics.compute_all(
        all_targets.flatten(),
        all_preds.flatten()
    )
    
    print("\nMétricas (escala normalizada):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    
    # Métricas en escala original
    preds_original = dataset.inverse_transform_target(all_preds)
    targets_original = dataset.inverse_transform_target(all_targets)
    
    metrics_original = SignalMetrics.compute_all(
        targets_original.flatten(),
        preds_original.flatten()
    )
    
    print("\nMétricas (escala original):")
    for name, value in metrics_original.items():
        print(f"  {name}: {value:.6f}")
    
    # Guardar métricas (convertir numpy floats a Python floats para JSON)
    def to_python_float(d):
        return {k: float(v) for k, v in d.items()}
    
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'normalized': to_python_float(metrics),
            'original_scale': to_python_float(metrics_original)
        }, f, indent=2)
    
    # =========================================================================
    # Visualizaciones
    # =========================================================================
    print("\n" + "-"*60)
    print("GENERANDO VISUALIZACIONES")
    print("-"*60)
    
    # Historia de entrenamiento
    fig = plot_training_history(history, save_path=os.path.join(exp_dir, 'training_history.png'))
    
    # Predicciones multi-step (algunos ejemplos)
    for idx in [0, len(all_preds)//2, -1]:
        # Obtener contexto para visualización
        enc_sample, dec_sample = None, None
        for (enc_in, dec_in), _ in test_ds.take(1):
            enc_sample = enc_in[0, :, 0].numpy()  # Primera feature del contexto
            break
        
        if enc_sample is not None:
            plot_multistep_prediction(
                context=enc_sample,
                y_true=targets_original[idx, :, 0],
                y_pred=preds_original[idx, :, 0],
                sample_idx=idx,
                save_path=os.path.join(exp_dir, f'prediction_sample_{idx}.png')
            )
    
    # =========================================================================
    # Gráfica comparativa de señal completa
    # =========================================================================
    print("\n" + "-"*60)
    print("GENERANDO COMPARACIÓN DE SEÑAL COMPLETA")
    print("-"*60)
    
    # Comparación en conjunto de entrenamiento
    fig_train, metrics_train = plot_full_signal_comparison(
        model=model,
        dataset=dataset,
        split='train',
        save_path=os.path.join(exp_dir, 'full_signal_comparison_train.png')
    )
    
    # Comparación en conjunto de test
    fig_test, metrics_test = plot_full_signal_comparison(
        model=model,
        dataset=dataset,
        split='test',
        save_path=os.path.join(exp_dir, 'full_signal_comparison_test.png')
    )
    
    # =========================================================================
    # Resumen final
    # =========================================================================
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nResultados guardados en: {exp_dir}")
    print(f"  - config.json: Configuración del experimento")
    print(f"  - metrics.json: Métricas de evaluación")
    print(f"  - checkpoints/: Pesos del modelo")
    print(f"  - logs/: TensorBoard logs")
    print(f"  - *.png: Visualizaciones")
    print(f"\nPara ver TensorBoard: tensorboard --logdir {log_dir}")
    
    return model, dataset, history


if __name__ == '__main__':
    model, dataset, history = main()