import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import datetime
import argparse
from src.models import create_model


def train_model(train_set, test_set, args):
  if args.load_path == None:
    model = create_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
  else:
    model = models.load_model(args.load_path) 

  model.summary()

  model_path = './models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  logs_path  = model_path + "/logs"
  best_model_path = model_path + "/best"
  checkpoint_path = model_path + "/checkpoints/cp-{epoch:04d}.ckpt"

  # Callbacks - saving info during training
  # Saving logs for Tensorboard
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path,
                                                        update_freq=args.logs_freq)
  # Saving recent model to continue training if interrupted
  checkpoint_callback  = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
                                                            save_weights_only=False,
                                                            save_freq=args.check_len)
  # Saving best model to evaluate
  best_model_callback  = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, 
                                                            save_weights_only=True,
                                                            monitor='accuracy',   # Should be 'val_accuracy', but doesn't work
                                                            save_best_only=True,
                                                            save_freq=args.check_len)

  history = model.fit(train_set,
                      epochs=args.start_epoch + args.num_epoch,
                      callbacks=[tensorboard_callback, checkpoint_callback, best_model_callback],
                      initial_epoch=args.start_epoch,
                      steps_per_epoch=args.epoch_len,
                      validation_data=test_set,
                      validation_steps=args.valid_len)
