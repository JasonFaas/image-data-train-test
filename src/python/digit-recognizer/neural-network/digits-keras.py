from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping

from image_test_space import DisplayImage


resources = '../../../resources/digit-recognizer'

# test_csv = 'test.csv'
test_csv = 'jason_test_1000.csv'

# train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
train_csv = 'jason_train_5000.csv'
# train_csv = 'jason_train_4000.csv'
# train_csv = 'jason_train_2000.csv'
# train_csv = 'jason_train_1000.csv'

train_csv_filename = '%s/%s' % (resources, train_csv)
test_csv_filename = '%s/%s' % (resources, test_csv)

image_mod = DisplayImage()
x_train, x_test, y_train, y_test = image_mod.train_test_set(train_file=train_csv_filename,
                                                        train_size=.8,
                                                        random_state=10)
# x_train, x_test, y_train, y_test = DisplayImage().train_test_set(train_file=train_csv_filename,
#                                                                  train_size=1.0,
#                                                                  test_file=test_csv_filename)


model_scaler = StandardScaler()
x_train_v2 = model_scaler.fit_transform(x_train)
x_test_v2 = model_scaler.transform(x_test)

y_train_v2 = to_categorical(y_train)


# learning_rates = [.0001, 0.01, 1]
# for lr in learning_rates:
# Create the model: model

layer_sizes = [100]#, .1]
layers = [1]
dataframe = pd.DataFrame(data=np.zeros((len(layers), len(layer_sizes)), dtype=np.float), columns=layer_sizes, index=layers)
print(dataframe)

for layer_size in layer_sizes:
    for layer_count in layers:
        model = Sequential()
        model.add(Dense(layer_size, activation='relu', input_shape=(784,)))
        for layer in range(0, layer_count):
            model.add(Dense(layer_size, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        early_stopping_monitor = EarlyStopping(patience=10, monitor='acc')
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(x_train_v2, y_train_v2, epochs=50, callbacks=[early_stopping_monitor], verbose=0)

        predictions = model.predict(x_test_v2)
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
