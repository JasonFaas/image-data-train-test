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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


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

x_train, x_test, y_train, y_test = train_test_split(samples_v2, target, test_size=0.2, random_state=10)

model_scaler = StandardScaler()
x_train_v2 = model_scaler.fit_transform(x_train)
x_test_v2 = model_scaler.transform(x_test)



review_failures = False

print(len(x_train_v2))

parameters = {'loss':['log', 'hinge'], 'alpha':[0.001,0.1,10], 'penalty':['l1', 'l2']}
linear_classifier = SGDClassifier(random_state=0)

from sklearn.model_selection import GridSearchCV
model_grid = GridSearchCV(linear_classifier, parameters, cv=4)
model_grid.fit(x_train_v2, y_train)
print("Best CV params", model_grid.best_params_)
print("Best CV accuracy", model_grid.best_score_)
print("Test accuracy of best grid search hypers:", model_grid.score(x_test_v2, y_test))

