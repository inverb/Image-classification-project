# Image classification project

In this project we explore using deep neural networks to various image classification tasks from Tensorflow/datasets.

## Team
- Mateusz Basiak (choosing models, training, tuning hyperparameters)
- Adrianna Struzik (data preparation, documentation, repo mainentance)

## Tools

- Python 3.7
- TensorFlow 2.0
- Google Collab
- Github
- etc.

## Project Structure

    .
    ├── config/                 # Config files (.yml, .json)
    ├── data/                   # dataset path
    ├── docs/                   # notebooks with project reports
    ├── models/                 # links to best models and their training logs
    ├── scripts/                # scripts used for training and evaluation
    └── src/               		# source code of the training and models

## Dataset

We will work with [food101](https://www.tensorflow.org/datasets/catalog/food101) dataset that consists of 101 000 images of 101 different kinds of food. We plan to start with some small subset of labels (10 to 20) and test our model before launching it on full dataset. 

Data is resized to 512x512 pixels, but it is not previously cleaned and possibly incorrectly labeled. It is divided into training set (3/4 of entire dataset) and test set.
