import sys
import tensorflow.keras
import tensorflow as tf

def main ():
    print(f"Enviroment: ")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras}")
    print(f"Python {sys.version}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

if __name__ == '__main__':
    main()