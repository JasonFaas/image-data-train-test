import numpy as np
import cv2 as cv
import operator
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

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

mergings = linkage(samples_v2, method='complete')
dendrogram(mergings, labels=target, leaf_rotation=90, leaf_font_size=6)
plt.show()
