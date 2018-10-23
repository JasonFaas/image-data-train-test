from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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

x_train, x_test, y_train, y_test = train_test_split(samples_v2, target, test_size=0.2, random_state=10)

model_scaler = StandardScaler()
x_train_v2 = model_scaler.fit_transform(x_train)
x_test_v2 = model_scaler.transform(x_test)

y_train_v2 = to_categorical(y_train)

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# learning_rates = [.0001, 0.01, 1]
# for lr in learning_rates:
# Create the model: model
model = Sequential()
# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))
# Add the second hidden layer
model.add(Dense(50, activation='relu'))
# Add the output layer
model.add(Dense(10, activation='softmax'))
# Compile the model
early_stopping_monitor = EarlyStopping(patience=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model
model.fit(x_train_v2, y_train_v2, validation_split=0.2, epochs=20, callbacks=[early_stopping_monitor])


print("\n\nStarting the Real Stuff")
predictions = model.predict(x_test_v2)
print(list(map(lambda v: np.argmin(np.array(v)), predictions)))
# array = np.array(predictions[12])
# print(array)
# print(array.argmax())

# print(model.summary())

