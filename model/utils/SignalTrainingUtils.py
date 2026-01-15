"""
Utilidades para entrenamiento de Signal Transformer
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import os

class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule del paper "Attention Is All You Need".
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step + 1)  # +1 para evitar división por cero
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            'd_model': int(self.d_model.numpy()),
            'warmup_steps': self.warmup_steps
        }


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay con warmup lineal.
    Alternativa más moderna al schedule original.
    """
    
    def __init__(
        self, 
        initial_lr: float = 1e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 50000,
        min_lr: float = 1e-6
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Warmup lineal
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # Cosine decay
        progress = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + tf.cos(np.pi * progress))
        
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
    
    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'warmup_steps': self.warmup_steps,
            'decay_steps': self.decay_steps,
            'min_lr': self.min_lr
        }


def masked_mse_loss(y_true, y_pred):
    """MSE loss (sin máscara, para señales continuas)."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def masked_mae_loss(y_true, y_pred):
    """MAE loss."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))


class SignalMetrics:
    """Métricas para evaluación de predicción de señales."""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coeficiente de determinación R²."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calcula todas las métricas."""
        return {
            'MSE': SignalMetrics.mse(y_true, y_pred),
            'RMSE': SignalMetrics.rmse(y_true, y_pred),
            'MAE': SignalMetrics.mae(y_true, y_pred),
            'MAPE': SignalMetrics.mape(y_true, y_pred),
            'R2': SignalMetrics.r2_score(y_true, y_pred)
        }


def create_callbacks(
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    patience: int = 10,
    model_name: str = 'signal_transformer',
    use_lr_schedule: bool = True
) -> list:
    """
    Crea callbacks estándar para entrenamiento.
    
    Args:
        checkpoint_dir: Directorio para guardar checkpoints
        log_dir: Directorio para logs de TensorBoard
        patience: Épocas sin mejora antes de early stopping
        model_name: Nombre base para archivos
        use_lr_schedule: Si True, no incluye ReduceLROnPlateau (incompatible con schedules)
    
    Returns:
    --------
    list de callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        # Guardar mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, f'{model_name}_best.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ),
    ]
    
    # Solo agregar ReduceLROnPlateau si NO se usa un schedule personalizado
    if not use_lr_schedule:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
    
    return callbacks


def plot_training_history(history, save_path: Optional[str] = None):
    """Grafica historial de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # MAE si existe
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time: Optional[np.ndarray] = None,
    title: str = 'Predictions vs Ground Truth',
    save_path: Optional[str] = None,
    num_samples: int = 500
):
    """
    Grafica predicciones vs valores reales.
    
    Parámetros:
    -----------
    y_true : array de valores reales
    y_pred : array de predicciones
    time : array de tiempo (opcional)
    num_samples : número de muestras a mostrar
    """
    # Limitar muestras para visualización
    n = min(len(y_true), num_samples)
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    
    if time is None:
        time = np.arange(n)
    else:
        time = time[:n]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Señales superpuestas
    axes[0].plot(time, y_true, label='Ground Truth', linewidth=1.5, alpha=0.8)
    axes[0].plot(time, y_pred, label='Prediction', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error
    error = y_true - y_pred
    axes[1].plot(time, error, color='red', linewidth=1, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[1].fill_between(time, error, 0, alpha=0.3, color='red')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Error')
    axes[1].set_title(f'Prediction Error (MAE: {np.mean(np.abs(error)):.6f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_multistep_prediction(
    context: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Visualiza una predicción multi-step individual.
    
    Parámetros:
    -----------
    context : ventana de contexto (input_window,)
    y_true : valores reales futuros (output_window,)
    y_pred : predicciones (output_window,)
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Contexto
    context_time = np.arange(len(context))
    ax.plot(context_time, context, 'b-', linewidth=1.5, label='Context (Input)', alpha=0.7)
    
    # Predicción
    pred_time = np.arange(len(context), len(context) + len(y_pred))
    ax.plot(pred_time, y_true, 'g-', linewidth=2, label='Ground Truth', marker='o', markersize=4)
    ax.plot(pred_time, y_pred, 'r--', linewidth=2, label='Prediction', marker='x', markersize=6)
    
    # Línea vertical separadora
    ax.axvline(x=len(context) - 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(f'Multi-step Prediction (Sample {sample_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_full_signal_comparison(
    model,
    dataset,
    split: str = 'train',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Genera predicciones sobre todo el dataset y las compara con la señal real.
    
    Reconstruye la señal completa usando ventanas deslizantes sin solapamiento,
    tomando solo el primer valor predicho de cada ventana para evitar discontinuidades.
    
    Parámetros:
    -----------
    model : SignalTransformer entrenado
    dataset : SignalDataset con los datos
    split : 'train', 'val', o 'test'
    save_path : ruta para guardar la figura
    figsize : tamaño de la figura
    
    Returns:
    --------
    fig : figura de matplotlib
    metrics : diccionario con métricas de la reconstrucción
    """
    # Seleccionar datos según split
    if split == 'train':
        features = dataset.train_features
        target = dataset.train_target
    elif split == 'val':
        features = dataset.val_features
        target = dataset.val_target
    elif split == 'test':
        features = dataset.test_features
        target = dataset.test_target
    else:
        raise ValueError(f"split debe ser 'train', 'val', o 'test', no '{split}'")
    
    input_window = dataset.input_window
    output_window = dataset.output_window
    
    # Generar predicciones con stride = output_window para reconstruir sin solapamiento
    predictions = []
    ground_truth = []
    positions = []  # Para tracking de posición temporal
    
    print(f"\nGenerando predicciones sobre {split} set...")
    print(f"  Total muestras: {len(features)}")
    print(f"  Input window: {input_window}")
    print(f"  Output window: {output_window}")
    
    # Recorrer con stride = output_window
    n_windows = 0
    for i in range(0, len(features) - input_window - output_window + 1, output_window):
        # Preparar entrada
        enc_input = features[i:i + input_window]
        enc_input = enc_input[np.newaxis, ...]  # (1, input_window, features)
        
        # Decoder input (teacher forcing con último valor conocido)
        dec_input = target[i + input_window - 1:i + input_window + output_window - 1]
        dec_input = dec_input[np.newaxis, :, np.newaxis]  # (1, output_window, 1)
        
        # Predicción
        pred = model((enc_input, dec_input), training=False)
        pred = pred.numpy().squeeze()  # (output_window,)
        
        # Ground truth correspondiente
        gt = target[i + input_window:i + input_window + output_window]
        
        predictions.extend(pred)
        ground_truth.extend(gt)
        positions.extend(range(i + input_window, i + input_window + output_window))
        n_windows += 1
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    positions = np.array(positions)
    
    print(f"  Ventanas procesadas: {n_windows}")
    print(f"  Puntos reconstruidos: {len(predictions)}")
    
    # Convertir a escala original
    pred_original = dataset.inverse_transform_target(predictions.reshape(-1, 1)).flatten()
    gt_original = dataset.inverse_transform_target(ground_truth.reshape(-1, 1)).flatten()
    
    # Señal original completa para referencia
    full_target = dataset.inverse_transform_target(target.reshape(-1, 1)).flatten()
    
    # Calcular métricas
    metrics = SignalMetrics.compute_all(gt_original, pred_original)
    
    # Crear figura
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Crear vector de tiempo (asumiendo 10kHz como en el dataset original)
    dt = 1e-4  # 10kHz
    time_full = np.arange(len(full_target)) * dt
    time_pred = positions * dt
    
    # Panel 1: Señal completa con predicciones superpuestas
    axes[0].plot(time_full, full_target, 'b-', linewidth=0.8, alpha=0.7, label='Ground Truth')
    axes[0].plot(time_pred, pred_original, 'r-', linewidth=0.8, alpha=0.8, label='Transformer Prediction')
    axes[0].set_xlabel('Tiempo [s]')
    axes[0].set_ylabel(f'{dataset.target_feature}')
    axes[0].set_title(f'Comparación Señal Completa ({split.upper()}) - R²={metrics["R2"]:.4f}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Zoom a una sección (primeros 10% de la señal)
    zoom_end = len(positions) // 10
    if zoom_end > 100:
        axes[1].plot(time_pred[:zoom_end], gt_original[:zoom_end], 'b-', linewidth=1.2, 
                     alpha=0.8, label='Ground Truth')
        axes[1].plot(time_pred[:zoom_end], pred_original[:zoom_end], 'r--', linewidth=1.2, 
                     alpha=0.9, label='Prediction')
        axes[1].set_xlabel('Tiempo [s]')
        axes[1].set_ylabel(f'{dataset.target_feature}')
        axes[1].set_title('Zoom: Primeros 10% de la señal reconstruida')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Error a lo largo del tiempo
    error = gt_original - pred_original
    axes[2].plot(time_pred, error, 'purple', linewidth=0.5, alpha=0.7)
    axes[2].fill_between(time_pred, error, 0, alpha=0.3, color='purple')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].axhline(y=np.mean(error) + 2*np.std(error), color='red', linestyle='--', 
                    linewidth=1, alpha=0.7, label=f'±2σ ({2*np.std(error):.6f})')
    axes[2].axhline(y=np.mean(error) - 2*np.std(error), color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].set_xlabel('Tiempo [s]')
    axes[2].set_ylabel('Error')
    axes[2].set_title(f'Error de Predicción - MAE={metrics["MAE"]:.6f}, RMSE={metrics["RMSE"]:.6f}')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Agregar texto con métricas
    metrics_text = (f'MSE: {metrics["MSE"]:.6f}\n'
                    f'RMSE: {metrics["RMSE"]:.6f}\n'
                    f'MAE: {metrics["MAE"]:.6f}\n'
                    f'MAPE: {metrics["MAPE"]:.2f}%\n'
                    f'R²: {metrics["R2"]:.4f}')
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figura guardada en: {save_path}")
    
    plt.show()
    
    return fig, metrics