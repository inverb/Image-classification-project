import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import datetime
import argparse

from src.data import load_classes_names, load_dataset, smaller_dataset
from src.training import train_model
from src.evaluation import evaluate_model


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', default='training', choices=['training', 'evaluation'], 
                      help='Decides whether training or evaluation will be performed. Defaults to training.')
  parser.add_argument('--load_path', default=None, help='''Link to path to load the model. Has to be provided in evaluation mode. 
                                                           In training mode leaving it empty will start training from scratch.''')
  parser.add_argument('--start_epoch', type=int, default=0, help='Number of starting epoch when restarting training.')
  parser.add_argument('--num_epoch', type=int, default=10, help='Number of epochs to train.')
  parser.add_argument('--epoch_len', type=int, default=None, 
                      help='Length of an epoch in batches. If None, epoch is iteration over whole dataset.')
  parser.add_argument('--logs_freq', type=int, default=1000, help='Time (in batches) between consecutive logs saves.')
  parser.add_argument('--check_len', type=int, default=100000, help='Time (in batches) between consecutive model saves.')
  parser.add_argument('--valid_len', type=int, default=None, 
                      help='Length (in batches) of validation. If None, validation is performed on whole validation set.')
  
  parser.add_argument('--batch_size', type=int, default=1, help='Changeable batch size.')
  parser.add_argument('--number_of_classes', type=int, default=101, 
                      help='Number of classes that we restrict our dataset to. Defaults to all classes.')
  
  parser.add_argument('--eval_num_samples', type=int, default=10, help='Number of samples we want to evaluate our model on.')
  args = parser.parse_args()


  # Loading dataset
  train_set, valid_set, test_set = load_dataset(batch_size)
  if number_of_classes < 101:
    train_set = smaller_dataset(train_set, number_of_classes)
    valid_set = smaller_dataset(valid_set, number_of_classes)
    test_set  = smaller_dataset(test_set,  number_of_classes)
    print("Loaded small dataset")
  else:
    print("Loaded big dataset")

  # Performing training or evaluation
  if args.mode == 'training':
    train_model(train_set, valid_set, args)
  else:
    labels_to_classes, classes_to_labels = load_classes_names()
    evaluate_model(test_set, labels_to_classes, classes_to_labels, args)
