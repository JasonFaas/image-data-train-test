# Mulit-class logistic regression:
# Train classifier on each label separately AND
# AND use those to predict

# Going to use Stratified Shuffle Split to verify that all target categories are in train data
# Modified to handle multiple categories


import numpy as np
import cv2 as cv
import operator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from image_test_space import DisplayImage
from sklearn.linear_model import LogisticRegression


resources = '../../../resources/digit-recognizer'

# Test files
test_csv = 'test.csv'
# test_csv = 'jason_test_1000.csv'

# Training files
train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
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
test_samples_v1 = digit_test_set.values

# separate training info into samples and target
train_samples_v1 = digit_train_set[:, 1]
target = digit_train_set[:, 0]
target = target.astype(int)






train_samples_v2 = list(map(lambda v: np.reshape(v, (-1)), train_samples_v1))
train_samples_v3 = image_info.circle_info_arr(train_samples_v2, train_samples_v1)


test_samples_v2 = list(map(lambda v: np.reshape(v, (28, -1, 1)), test_samples_v1))
test_samples_v2 = list(map(lambda v: v.astype(np.uint8), test_samples_v2))

test_samples_v3 = image_info.circle_info_arr(test_samples_v1, test_samples_v2)





from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(train_samples_v3)
x_test = scaler.transform(test_samples_v3)


clf = LogisticRegression(penalty='l2', C=.1)
clf.fit(x_train, target)

preds = clf.predict(x_test)

# put data into dataframe
result_df = pd.DataFrame(preds)
result_df.index += (1)
result_df = result_df.reset_index()
output_columns = ['ImageId', 'Label']
result_df.columns = output_columns

version = '1'
result_df.to_csv('submission-closed-info-logreg-v%s.csv' % version, index=False)
