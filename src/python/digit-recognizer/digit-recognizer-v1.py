#
# Basic v1 version is simple knn
#
#


import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from image_test_space import DisplayImage

# digit_train_set = pd.read_csv('../../resources/digit-recognizer/train.csv')
digit_train_set = pd.read_csv('../../resources/digit-recognizer/jason_train_5000.csv')



independent_columns = digit_train_set.columns[1:]
dependent_column = digit_train_set.columns[0:1]

x = digit_train_set.loc[:, independent_columns].values
y = digit_train_set.loc[:, dependent_column].values

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=91)

display_image = DisplayImage(data_set=x_test)

scores = {}
# creating loop for neighbor_itr for later
# 3 is best, though 1-10 are very similar at about 93.5%
for neighbor_itr in range(3, 4):
    knn = KNeighborsClassifier(n_neighbors=neighbor_itr)
    knn.fit(x_train, y_train.ravel())
    pred = knn.predict(x_test)
    y_test_raveled = y_test.ravel()
    score = accuracy_score(y_test_raveled, pred)
    tf_result = y_test_raveled == pred
    print(str(neighbor_itr) + ":" + str(score))
    for idx, tf in enumerate(tf_result):
        if not tf:
            print("Actual:" + str(y_test_raveled[idx]) + "::Guess:" + str(pred[idx]))
            if display_image.display_image(idx) == ord('q'):
                break


print("finished")
