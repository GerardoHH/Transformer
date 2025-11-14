import sys
import random
import numpy as np
import tensorflow as tf
import model.SimpleModelTrainUtils as smtu
import model.CustomModelTrainUtils as cmtu

def enviroment():
    print(f"Enviroment: ")
    print(f"\tPython {sys.version}")
    print(f"\tNumpy  {np.__version__}")
    print(f"\tTensor Flow Version: {tf.__version__}")
    print(f"\tKeras Version: {tf.keras.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    print("\tGPU is", "available" if gpu else "NOT AVAILABLE")
    
    print(f"\tEager execution: {tf.executing_eagerly()} ")
    print("-----------------------------------------------------------------")
    
def reproducibility():
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
def buildSimpleTranformer():
    smtu.build_and_train_simpleTrasformer()
    
def builCustomTransformer():
    
    cmtu.build_and_train_simpleTrasformer()
    
    