import matplotlib.pyplot as plt

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
