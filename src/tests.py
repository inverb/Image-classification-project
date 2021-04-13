import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow_datasets as tfds
import datetime
import argparse

# Testing if all images have the right size
def image_size_test(ds):
  images_h = [0 for _ in range(0, 1000)]
  images_w = [0 for _ in range(0, 1000)]
  total = 0
  for image, _ in ds:
    images_h[tf.shape(image)[1].numpy()] += 1
    images_w[tf.shape(image)[2].numpy()] += 1
    total += 1
  
  assert images_h[512] == total
  assert images_w[512] == total

# Testing if there's just one class the network outputs
def one_class_bias(test_set, labels_to_classes, classes_to_labels):
  with tf.device('/device:GPU:0'):
    model = create_model()
    model.load_weights(load_path).expect_partial()
    print("Model loaded from " + load_path)

    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    index = 0
    for (sample, label) in test_set:
      verdict = model.predict(sample)
      verdict_label = tf.math.argmax(tf.nn.softmax(verdict)[0]).numpy()
      correct_label = list(label.numpy())[0]
      if verdict_label == correct_label:
        index += 1
        print(labels_to_classes[verdict_label])
    print(index)
