import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# digit_train_set = pd.read_csv('../../resources/digit-recognizer/train.csv')
digit_train_set = pd.read_csv('../../resources/digit-recognizer/jason_1000.csv')
# digit_train_set = pd.read_csv('../../resources/digit-recognizer/jason_10000.csv')


independent_columns = digit_train_set.columns[1:]
dependent_column = digit_train_set.columns[0:1]

x = digit_train_set.loc[:, independent_columns].values
y = digit_train_set.loc[:, dependent_column].values

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=91)

scores = {}
# creating loop for neighbor_itr for later
# 3 is best, though 1-10 are very similar at about 93.5%
for neighbor_itr in range(3, 4):
    knn = KNeighborsClassifier(n_neighbors=neighbor_itr)
    knn.fit(x_train, y_train.ravel())
    pred = knn.predict(x_test)
    score = accuracy_score(y_test.ravel(), pred)
    print(str(neighbor_itr) + ":" + str(score))
    print(pred)


print("finished")
