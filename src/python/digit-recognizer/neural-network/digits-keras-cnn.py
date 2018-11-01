from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping

from image_test_space import DisplayImage


resources = '../../../resources/digit-recognizer/'

# test_csv = 'test.csv'
test_csv = 'jason_test_1000.csv'

train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
# train_csv = 'jason_train_5000.csv'
# train_csv = 'jason_train_4000.csv'
# train_csv = 'jason_train_2000.csv'
# train_csv = 'jason_train_1000.csv'

train_csv_filename = resources + train_csv
test_csv_filename = resources + test_csv

image_mod = DisplayImage()
x_train, x_test, y_train, y_test = image_mod.train_test_set(train_file=train_csv_filename,
                                                            train_size=.8,
                                                            random_state=10)
# x_train, x_test, y_train, y_test = image_mod.train_test_set(train_file=train_csv_filename,
#                                                                  train_size=1.0,
#                                                                  test_file=test_csv_filename)

# TODO investigate turning Scaler back on
# model_scaler = StandardScaler()
# x_train_v2 = model_scaler.fit_transform(x_train)
# x_test_v2 = model_scaler.transform(x_test)

x_train_v2 = list(map(lambda v: np.reshape(v, (28, 28, 1)), x_train))
x_test_v2 = list(map(lambda v: np.reshape(v, (28, 28, 1)), x_test))

y_train_v2 = to_categorical(y_train)


# learning_rates = [.0001, 0.01, 1]
# for lr in learning_rates:
# Create the model: model

layer_sizes = [100]#, .1]
layers = [1]
dataframe = pd.DataFrame(data=np.zeros((len(layers), len(layer_sizes)), dtype=np.float), columns=layer_sizes, index=layers)
print(dataframe)

input_shape = (28, 28, 1)

import tensorflow as tf
for layer_size in layer_sizes:
    for layer_count in layers:
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.softmax))

        # early_stopping_monitor = EarlyStopping(patience=10, monitor='acc')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # TODO review using y_train instead of y_train_v2 (aka to_categorical)
        model.fit(np.array(x_train_v2),
                  np.array(y_train),
                  epochs=50,
                  verbose=0)

        predictions = model.predict(np.array(x_test_v2))
        predictions = list(map(lambda v: np.argmax(np.array(v)), predictions))

        # scoring info
        if y_test is not None:
            total_test = y_test.shape[0]
            nonzero = np.count_nonzero(np.array(predictions == y_test))
            print("params:: layer_count: " + str(layer_count) + " " + "layer_size:" + str(layer_size))
            score = round(nonzero / total_test, 3)
            print("\tScore: " + str(score))
            dataframe.at[layer_count, layer_size] = score

print(dataframe)

image_mod.write_to_file(predictions)
