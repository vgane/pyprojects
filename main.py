# This is a sample Python script.




import tensorflow as tf
from tensorflow.python import keras
import  numpy as np
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images.shape



if __name__ == '__main__':
    print_hi('PyCharm')

    # Feature columns describe how to use the input.
