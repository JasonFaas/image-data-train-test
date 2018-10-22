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


model = PCA(n_components=600)
digit_features = model.fit_transform(samples_v2)


# graph of features
# features = range(model.n_components_)
# plt.bar(features, model.explained_variance_)
# plt.xticks(features)
# plt.yscale('log')
# plt.ylabel('variance')
# plt.xlabel('PCA feature')
# plt.show()
# NOTES: Reviewing above graph shows that about 600 of 784 features are relevant


# K Means Section
prev_inertia = 10 * 1000
for i in range(10,11):
    model_loop = KMeans(n_clusters=i)
    model_loop.fit(samples_v2)
    print("inertia " + str(i) + ":" + str(int(model_loop.inertia_)))
    print("\tDecrease %:" + str(round(100 - (100 * model_loop.inertia_ / prev_inertia), 2)))
    prev_inertia = model_loop.inertia_



model_k_means = KMeans(n_clusters=20)
model_k_means.fit(samples_v2)
pred_labels = model_k_means.predict(samples_v2)

print(pd.crosstab(pred_labels, target))