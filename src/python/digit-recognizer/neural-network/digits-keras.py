from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from image_test_space import DisplayImage

resources = '../../../resources/digit-recognizer'
# train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
train_csv = 'jason_train_5000.csv'
# train_csv = 'jason_train_4000.csv'
# train_csv = 'jason_train_2000.csv'
# train_csv = 'jason_train_1000.csv'
csv_filename = '%s/%s' % (resources, train_csv)

# read training info
digit_train_set = pd.read_csv(csv_filename)
image_info = DisplayImage(csv_filename)
digit_train_set = image_info.get_all_info()

# separate training info into samples and target
samples_v1 = digit_train_set[:, 1]
target = digit_train_set[:, 0]
target = target.astype('int')

samples_v2 = list(map(lambda v: np.reshape(v, (-1)), samples_v1))

x_train, x_test, y_train, y_test = train_test_split(samples_v2,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)

model_scaler = StandardScaler()
x_train_v2 = model_scaler.fit_transform(x_train)
x_test_v2 = model_scaler.transform(x_test)

y_train_v2 = to_categorical(y_train)

from keras.optimizers import SGD


# learning_rates = [.0001, 0.01, 1]
# for lr in learning_rates:
# Create the model: model

layer_sizes = [100, 150, 200]#, .1]
layers = [1, 2]
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

        total_test = y_test.shape[0]
        nonzero = np.count_nonzero(np.array(predictions == y_test))
        print("params:: layer_count: " + str(layer_count) + " " + "layer_size:" + str(layer_size))
        score = round(nonzero / total_test, 3)
        print("\tScore: " + str(score))
        dataframe.at[layer_count, layer_size] = score

print(dataframe)