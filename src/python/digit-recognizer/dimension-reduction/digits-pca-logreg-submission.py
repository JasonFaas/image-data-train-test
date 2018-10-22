import numpy as np
import cv2 as cv
import operator
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from image_test_space import DisplayImage

resources = '../../../resources/digit-recognizer'
test_csv = 'test.csv'
# test_csv = 'jason_test_1000.csv'
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
x_actual_test = digit_test_set.values

# separate training info into samples and target
samples_v1 = digit_train_set[:, 1]
target = digit_train_set[:, 0]
target = target.astype('int')

samples_v2 = list(map(lambda v: np.reshape(v, (-1)), samples_v1))


model_scaler = StandardScaler()
samples_v2 = model_scaler.fit_transform(samples_v2)
x_actual_test_scaled = model_scaler.transform(x_actual_test)

model_pca = PCA(0.9)
samples_v3 = model_pca.fit_transform(samples_v2)
x_actual_test_pca_tranform = model_pca.transform(x_actual_test_scaled)


logreg = LogisticRegression(solver='lbfgs')
logreg.fit(samples_v3, target)
preds = logreg.predict(x_actual_test_pca_tranform)
# print(logreg.score(x_test, y_test))


# put data into dataframe
result_df = pd.DataFrame(preds)
result_df.index += (1)
result_df = result_df.reset_index()
output_columns = ['ImageId', 'Label']
result_df.columns = output_columns

version = '2'
result_df.to_csv('submission-pca-logreg-v%s.csv' % version, index=False)