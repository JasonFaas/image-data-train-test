from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from image_test_space import DisplayImage


resources = '../../../resources/digit-recognizer'
test_csv = 'test.csv'
# test_csv = 'jason_test_1000.csv'
# train_csv = 'train.csv'
train_csv = 'jason_train_10000.csv'
# train_csv = 'jason_train_5000.csv'
# train_csv = 'jason_train_4000.csv'
# train_csv = 'jason_train_2000.csv'
# train_csv = 'jason_train_1000.csv'
train_csv_filename = '%s/%s' % (resources, train_csv)
test_csv_filename = '%s/%s' % (resources, test_csv)

# read training info
digit_train_set = pd.read_csv(train_csv_filename)
image_info = DisplayImage(train_csv_filename)
digit_train_set = image_info.get_all_info()
# read testing info
digit_test_set = pd.read_csv(test_csv_filename)
x_actual_test = digit_test_set.values

# separate training info into samples and target
samples_v1 = digit_train_set[:, 1]
target = digit_train_set[:, 0]
target = target.astype('int')

samples_v2 = list(map(lambda v: np.reshape(v, (-1)), samples_v1))


model_scaler = StandardScaler()
samples_v2 = model_scaler.fit_transform(samples_v2)
x_actual_test_scaled = model_scaler.transform(x_actual_test)

y_train_v2 = to_categorical(target)

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
early_stopping_monitor = EarlyStopping(patience=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model
model.fit(samples_v2, y_train_v2, validation_split=0.5, epochs=20, callbacks=[early_stopping_monitor])


print("\n\nStarting the Real Stuff")
predictions = model.predict(x_actual_test_scaled)
predictions_list = list(map(lambda v: np.argmax(np.array(v)), predictions))
print(predictions_list)

print(model.summary())

# Write to file
result_df = pd.DataFrame(predictions_list)
result_df.index += (1)
result_df = result_df.reset_index()
output_columns = ['ImageId', 'Label']
result_df.columns = output_columns

version = '2'
result_df.to_csv('submission-keras-v%s.csv' % version, index=False)

