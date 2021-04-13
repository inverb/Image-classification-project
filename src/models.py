import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import datetime
import argparse


def create_model():

  # model = models.Sequential()
  # model.add(layers.MaxPooling2D((4, 4), input_shape=(512, 512, 3)))
  # model.add(layers.Flatten())
  # model.add(layers.Dense(1000, activation='relu'))
  # model.add(layers.Dropout(0.5))
  # model.add(layers.Dense(number_of_classes))


  # Smaller
  # model = models.Sequential()
  # model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(512, 512, 3)))
  # model.add(layers.MaxPooling2D((3, 3), strides=2))
  # model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  # model.add(layers.Conv2D(16, (3, 3), activation='relu'))
  # model.add(layers.MaxPooling2D((3, 3), strides=2))
  # model.add(layers.Flatten())
  # model.add(layers.Dense(1000, activation='relu'))
  # model.add(layers.Dropout(0.5))
  # model.add(layers.Dense(number_of_classes))

  # Based on classic paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
  # Bigger
  model = models.Sequential()
  model.add(layers.Conv2D(16, (5, 5), strides=4, activation='relu',  input_shape=(512, 512, 3)))
  model.add(layers.MaxPooling2D((3, 3), strides=2))
  model.add(layers.Conv2D(64, (3, 3), strides=2, activation='relu' ))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((3, 3), strides=2))
  model.add(layers.Flatten())
  model.add(layers.Dense(1000, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1000, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(number_of_classes))    # normally 101, in small dataset around 10

  return model
