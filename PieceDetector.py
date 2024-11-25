import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical



def load_data():
    path = "./image_tiles"

def cnn():
    