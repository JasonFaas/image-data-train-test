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

from image_test_space import DisplayImage

resources = '../../../resources/digit-recognizer'
train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
# train_csv = 'jason_train_5000.csv'
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(samples_v2, target, test_size=0.2, random_state=91)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(x_train, y_train)
# preds = logreg.predict(x_test)
print(logreg.score(x_test, y_test))
