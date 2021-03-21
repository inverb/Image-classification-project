import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import datetime
import argparse


def create_model():
  # All generic, taken from here: https://www.tensorflow.org/tutorials/images/cnn
  # Here may be some ideas: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

  model = models.Sequential()
  model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(512, 512, 3)))
  model.add(layers.MaxPooling2D((3, 3)))
  model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((3, 3)))
  model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(1000, activation='relu'))
  model.add(layers.Dense(101))

  return model
