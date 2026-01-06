"""
Signal Dataset - Preprocesamiento parametrizable para series temporales
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SignalDataset:
    """
    Dataset para prediccion de senales con Transformer.    
    Parametros:
    -----------
    filepath : str
        Ruta al archivo CSV con los datos
    input_features : List[str]
        Lista de columnas a usar como features de entrada
    target_feature : str
        Columna objetivo a predecir
    input_window : int
        Tamano de la ventana de entrada (cuantos pasos históricos usar)
    output_window : int
        Tamano de la ventana de salida (cuantos pasos predecir)
    stride : int
        Paso entre ventanas consecutivas (default=1)
    train_split : float
        Proporcion de datos para entrenamiento (default=0.8)
    val_split : float
        Proporcion de datos para validacion (default=0.1)
    scaler_type : str
        Tipo de escalador: 'standard' o 'minmax' (default='standard')
    """
    
    def __init__(
        self,
        filepath: str,
        input_features: List[str],
        target_feature: str,
        input_window: int = 100,
        output_window: int = 10,
        stride: int = 1,
        train_split: float = 0.8,
        val_split: float = 0.1,
        scaler_type: str = 'standard'
    ):
        self.filepath = filepath
        self.input_features = input_features
        self.target_feature = target_feature
        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride
        self.train_split = train_split
        self.val_split = val_split
        self.scaler_type = scaler_type
        
        # Scalers para features y target (separados para poder invertir)
        self.feature_scaler = None
        self.target_scaler = None
        
        # Datos procesados
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Cargar y procesar
        self._load_data()
        self._create_scalers()
        self._split_data()
        
    def _load_data(self):
        """Carga el CSV y extrae las columnas relevantes."""
        df = pd.read_csv(self.filepath)
        
        # Verificar que existen las columnas
        all_features = self.input_features + [self.target_feature]
        missing = [f for f in all_features if f not in df.columns]
        if missing:
            raise ValueError(f"Columnas no encontradas: {missing}\nDisponibles: {list(df.columns)}")
        
        # Extraer features y target
        self.feature_data = df[self.input_features].values.astype(np.float32)
        self.target_data = df[self.target_feature].values.astype(np.float32).reshape(-1, 1)
        
        print(f"Datos cargados: {len(df)} muestras")
        print(f"Features de entrada ({len(self.input_features)}): {self.input_features}")
        print(f"Target: {self.target_feature}")
        print(f"Ventana entrada: {self.input_window}, Ventana salida: {self.output_window}")
        
    def _create_scalers(self):
        """Crea y ajusta los scalers."""
        if self.scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"scaler_type debe ser 'standard' o 'minmax', no '{self.scaler_type}'")
        
        # Ajustar scalers con todos los datos (antes de split para consistencia)
        self.feature_scaler.fit(self.feature_data)
        self.target_scaler.fit(self.target_data)
        
        # Escalar datos
        self.feature_data_scaled = self.feature_scaler.transform(self.feature_data)
        self.target_data_scaled = self.target_scaler.transform(self.target_data).flatten()
        
    def _split_data(self):
        """Divide los datos en train/val/test."""
        n = len(self.feature_data_scaled)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))
        
        self.train_features = self.feature_data_scaled[:train_end]
        self.train_target = self.target_data_scaled[:train_end]
        
        self.val_features = self.feature_data_scaled[train_end:val_end]
        self.val_target = self.target_data_scaled[train_end:val_end]
        
        self.test_features = self.feature_data_scaled[val_end:]
        self.test_target = self.target_data_scaled[val_end:]
        
        print(f"\nSplit de datos:")
        print(f"  Train: {len(self.train_features)} muestras")
        print(f"  Val:   {len(self.val_features)} muestras")
        print(f"  Test:  {len(self.test_features)} muestras")
        
    def _create_sequences(
        self, 
        features: np.ndarray, 
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crea secuencias de entrada y salida para el modelo.
        
        Returns:
        --------
        encoder_input : (N, input_window, num_features)
            Ventana de contexto con todas las features
        decoder_input : (N, output_window, 1)
            Target desplazado (teacher forcing)
        decoder_target : (N, output_window, 1)
            Valores a predecir
        """
        total_window = self.input_window + self.output_window
        n_sequences = (len(features) - total_window) // self.stride + 1
        
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []
        
        for i in range(0, len(features) - total_window + 1, self.stride):
            # Encoder: ventana de entrada con todas las features
            enc_in = features[i:i + self.input_window]
            
            # Decoder input: target desplazado (para teacher forcing)
            # Incluye el último valor conocido + los primeros output_window-1 valores objetivo
            dec_in = target[i + self.input_window - 1:i + self.input_window + self.output_window - 1]
            
            # Decoder target: los valores a predecir
            dec_target = target[i + self.input_window:i + self.input_window + self.output_window]
            
            encoder_inputs.append(enc_in)
            decoder_inputs.append(dec_in)
            decoder_targets.append(dec_target)
            
        return (
            np.array(encoder_inputs, dtype=np.float32),
            np.array(decoder_inputs, dtype=np.float32)[..., np.newaxis],
            np.array(decoder_targets, dtype=np.float32)[..., np.newaxis]
        )
    
    def get_train_dataset(self, batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
        """Retorna tf.data.Dataset para entrenamiento."""
        enc_in, dec_in, dec_target = self._create_sequences(self.train_features, self.train_target)
        
        dataset = tf.data.Dataset.from_tensor_slices(((enc_in, dec_in), dec_target))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(enc_in))
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"\nDataset de entrenamiento:")
        print(f"  Secuencias: {len(enc_in)}")
        print(f"  Encoder input shape: {enc_in.shape}")
        print(f"  Decoder input shape: {dec_in.shape}")
        print(f"  Target shape: {dec_target.shape}")
        
        return dataset
    
    def get_val_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Retorna tf.data.Dataset para validación."""
        enc_in, dec_in, dec_target = self._create_sequences(self.val_features, self.val_target)
        
        dataset = tf.data.Dataset.from_tensor_slices(((enc_in, dec_in), dec_target))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_test_dataset(self, batch_size: int = 32) -> tf.data.Dataset:
        """Retorna tf.data.Dataset para test."""
        enc_in, dec_in, dec_target = self._create_sequences(self.test_features, self.test_target)
        
        dataset = tf.data.Dataset.from_tensor_slices(((enc_in, dec_in), dec_target))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """Convierte valores escalados a escala original."""
        shape = scaled_values.shape
        flat = scaled_values.reshape(-1, 1)
        original = self.target_scaler.inverse_transform(flat)
        return original.reshape(shape)
    
    def get_config(self) -> dict:
        """Retorna configuración del dataset."""
        return {
            'input_features': self.input_features,
            'target_feature': self.target_feature,
            'input_window': self.input_window,
            'output_window': self.output_window,
            'stride': self.stride,
            'num_features': len(self.input_features),
            'scaler_type': self.scaler_type
        }


# Funcion de conveniencia para crear dataset rapidamente
def create_signal_dataset(
    filepath: str,
    input_features: List[str],
    target_feature: str,
    input_window: int = 100,
    output_window: int = 10,
    batch_size: int = 32,
    **kwargs
) -> Tuple[SignalDataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Crea dataset y retorna los tf.data.Dataset listos para usar.
    
    Returns:
    --------
    dataset : SignalDataset
        Objeto dataset con métodos y scalers
    train_ds, val_ds, test_ds : tf.data.Dataset
        Datasets listos para entrenamiento
    """
    dataset = SignalDataset(
        filepath=filepath,
        input_features=input_features,
        target_feature=target_feature,
        input_window=input_window,
        output_window=output_window,
        **kwargs
    )
    
    train_ds = dataset.get_train_dataset(batch_size=batch_size)
    val_ds = dataset.get_val_dataset(batch_size=batch_size)
    test_ds = dataset.get_test_dataset(batch_size=batch_size)
    
    return dataset, train_ds, val_ds, test_ds
