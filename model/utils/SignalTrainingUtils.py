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
        arg1 = tf.math.rsqrt(step + 1)  # +1 para evitar divisi�n por cero
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
    Alternativa mas moderna al schedule original.
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
    """MSE loss (sin m�scara, para se�ales continuas)."""
    return tf.reduce_mean(tf.square(y_true - y_pred))


def masked_mae_loss(y_true, y_pred):
    """MAE loss."""
    return tf.reduce_mean(tf.abs(y_true - y_pred))


class SignalMetrics:
    """M�tricas para evaluaci�n de predicci�n de se�ales."""
    
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
        """Coeficiente de determinaci�n R�."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calcula todas las m�tricas."""
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
    model_name: str = 'signal_transformer'
) -> list:
    """
    Crea callbacks estandar para entrenamiento.
    
    Returns:
    --------
    list de callbacks: [ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau]
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
        
        # Reducir LR si no mejora (�til si no usas schedule personalizado)
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
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
    
    Par�metros:
    -----------
    y_true : array de valores reales
    y_pred : array de predicciones
    time : array de tiempo (opcional)
    num_samples : n�mero de muestras a mostrar
    """
    # Limitar muestras para visualizaci�n
    n = min(len(y_true), num_samples)
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    
    if time is None:
        time = np.arange(n)
    else:
        time = time[:n]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Se�ales superpuestas
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
    Visualiza una predicci�n multi-step individual.
    
    Par�metros:
    -----------
    context : ventana de contexto (input_window,)
    y_true : valores reales futuros (output_window,)
    y_pred : predicciones (output_window,)
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Contexto
    context_time = np.arange(len(context))
    ax.plot(context_time, context, 'b-', linewidth=1.5, label='Context (Input)', alpha=0.7)
    
    # Predicci�n
    pred_time = np.arange(len(context), len(context) + len(y_pred))
    ax.plot(pred_time, y_true, 'g-', linewidth=2, label='Ground Truth', marker='o', markersize=4)
    ax.plot(pred_time, y_pred, 'r--', linewidth=2, label='Prediction', marker='x', markersize=6)
    
    # L�nea vertical separadora
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
