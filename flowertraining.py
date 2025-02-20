# This is a sample Python script.

import pandas as pd


import tensorflow as tf
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    train_path = tf.keras.utils.get_file(
        "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    test_path = tf.keras.utils.get_file(
        "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

    train_y=train.pop('Species')
    test_y=test.pop('Species')
    print(train.head())
  #  print(train_y.head())

    # def input_fn(features, labels, training=True, batch_size=256):
    #     """An input function for training or evaluating"""
    #     # Convert the inputs to a Dataset.
    #     dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    #
    #     # Shuffle and repeat if you are in training mode.
    #     if training:
    #         dataset = dataset.shuffle(1000).repeat()
    #
    #     return dataset.batch(batch_size)
    #
    my_feature_columns = []
    for key in train.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))


    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
            # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
            # The model must choose between 3 classes.
    n_classes=3)
    #
    #
    #
    # # Feature columns describe how to use the input.
    # my_feature_columns = []
    # for key in train.keys():
    #     my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    #
    # classifier = tf.estimator.DNNClassifier(
    # feature_columns=my_feature_columns,
    # hidden_units=[30,10],
    # n_classes=3)
    #
    # # Train the Model.
    # classifier.train(
    #     input_fn=lambda: input_fn(train, train_y, training=True),
    #     steps=5000)
    #
    # eval_result = classifier.evaluate(
    #     input_fn=lambda: input_fn(test, test_y, training=False))
    #
    # print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    #
    # # x= lambda :print("hi")
    #
    def input_fn2(features, labels, training=True, batch_size=256):
        """An input function for training or evaluating"""
        # Convert the inputs to a Dataset.
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
    features=['SepalLength','SepalWidth','PetalLength','PetalWidth']
    predict={}

    print("Please type numeric values as prompted")
    for feature in features:
        valid = True
        while valid:
         val = input(feature + ": ")
         if not val.isdigit(): valid = False

        predict[feature] = [float(val)]

    predictions = classifier.predict(input_fn2=lambda: input_fn2(predict))
    for pred_dict in predictions:
        print(pred_dict)
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print('Prediction 2 is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))


if __name__ == '__main__':
    print_hi('PyCharm')

    # Feature columns describe how to use the input.
