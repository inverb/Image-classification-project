import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import datetime
import argparse
from src.models import create_model


def evaluate_model(test_set, labels_to_classes, classes_to_labels, args):
  model_eval = create_model()
  model_eval.load_weights(args.load_path).expect_partial()
  print("Model loaded from " + args.load_path)

  model_eval.compile(optimizer='adam', 
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
  test_loss, test_acc = model_eval.evaluate(test_set)
  print("Test set accuracy: {}\nTest set loss: {}".format(test_acc, test_loss))

  # Drawing plots with model guesses. Unneccessary.
  # fig, axs = plt.subplots(int(eval_num_samples / 4), 4, figsize=(16, 8))
  # samples = small_train_set.shuffle(1000).take(eval_num_samples)
  # index = 0
  # for (sample, _) in samples:
  #   verdict = model_eval.predict(sample)
  #   verdict_label = tf.math.argmax(tf.nn.softmax(verdict)[0]).numpy()
  #   image = list(sample.numpy())[0]   # sample.as_numpy_iterator()

  #   axs[int(index/4), index%4].imshow(image)
  #   axs[int(index/4), index%4].set_title(labels_to_classes[verdict_label] 
  #                                        + '(' + str(verdict_label) + ')')
  #   axs[int(index/4), index%4].axis('off')
  #   index += 1
