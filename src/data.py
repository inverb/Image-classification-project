import tensorflow as tf
import tensorflow_datasets as tfds


def load_dataset(batch_size=1):
  # Dataset from here: https://www.tensorflow.org/datasets/catalog/food101
  # Based on: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf
  ds, ds_info = tfds.load('food101', 
                          split=['train', 'validation'], # loads both traing and validation sets
                          batch_size=batch_size,         # divides input into batches to process input simultaneously
                          as_supervised=True,            # loads dataset, each image is a tuple (image, label)
                          shuffle_files=True,           # shuffles images between epochs
                          with_info=True)                # loads (ds, ds_info) instead of plain ds

  def dataset_resize(ds):
    return ds.map(lambda image, label: (tf.image.resize(image, [512,512]), label))

  def separate_test_set(ds):
    valid_set = ds.skip(5050)
    test_set = ds.take(5050)
    return (valid_set, test_set)

  train_set = dataset_resize(ds[0])
  valid_set, test_set = separate_test_set(dataset_resize(ds[1]))
  return(train_set, valid_set, test_set)


# Creates smaller dataset with number_of_classes distinct labels
def smaller_dataset(dataset, number_of_classes=10):
  return ds.filter(lambda image, label:(label[0] < number_of_classes))


def load_classes_names():
  # Load names of classes
  with open('./data/classes.txt') as f:
      plik = f.readlines()
  labels_to_classes = [x.strip() for x in plik] 

  # Create dict from classes names to labels
  classes_to_labels = {}
  for i in range(0, 101):
    classes_to_labels[labels_to_classes[i]] =  i;

  assert classes_to_labels['tiramisu'] == 98

  return labels_to_classes, classes_to_labels
