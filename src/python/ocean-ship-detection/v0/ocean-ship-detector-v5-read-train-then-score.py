import numpy as np
import cv2 as cv
import pandas as pd
from image_modifications import ImageModifications
import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


resources = '../../resources/ocean-ship-detection/'
sub_image_dtype = {'ship_in_image': bool, 'blue_avg': np.double, 'green_avg': np.double, 'red_avg': np.double}
train_set = pd.read_csv(resources + 'jason_top_level_64_08.csv', dtype=sub_image_dtype)
test_set = pd.read_csv(resources + 'jason_top_level_64_09.csv', dtype=sub_image_dtype)

x_columns = ['blue_avg', 'green_avg', 'red_avg']
y_column = ['ship_in_image']
train_x_list = train_set[x_columns].values
train_y_list = train_set[y_column].values
test_x_list = test_set[x_columns].values
test_y_list = test_set[y_column].values



for neighbor_itr in range(1, 10):
    print("neighbors:" + str(neighbor_itr))
    knn = KNeighborsClassifier(n_neighbors=neighbor_itr)
    knn.fit(train_x_list, train_y_list.ravel())
    pred = knn.predict(test_x_list)
    y_test_raveled = test_y_list.ravel()
    score = accuracy_score(y_test_raveled, pred)
    tf_result = y_test_raveled == pred
    print(str(neighbor_itr) + ":" + str(int(100 * score)) + "%")



print("\nfinished")
