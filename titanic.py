# This is a sample Python script.
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
# import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                           'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
   # print(feature_columns)
    # #print(dftrain["embark_town"].unique())
    def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
        def input_function():
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.batch(batch_size).repeat(num_epochs)
            return ds

        return input_function

    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()
    # print(result['accuracy'])
    result= list(linear_est.predict(eval_input_fn))
    print(result[0]['probabilities'][1])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain.head()
