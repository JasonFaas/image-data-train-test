import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from image_test_space import DisplayImage

csv_filename = '../../resources/digit-recognizer/jason_train_1000.csv'
# csv_filename = '../../resources/digit-recognizer/train.csv'
digit_train_set = pd.read_csv(csv_filename)

image_info = DisplayImage(csv_filename)
digit_train_set = image_info.get_all_info()

digit_train_set[:, 1] = list(map(lambda v: cv.threshold(v, 175, 255, cv.THRESH_BINARY)[1], digit_train_set[:, 1]))
digit_train_set[:, 1] = list(map(lambda v: cv.dilate(v, np.ones((1,3),dtype=np.uint8)), digit_train_set[:, 1]))

x = digit_train_set[:,1]
y = digit_train_set[:,0]
y = y.astype('int')

x = list(map(lambda v: np.reshape(v, (-1)), x))
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
    y_test_raveled = y_test.ravel()
    score = accuracy_score(y_test_raveled, pred)
    tf_result = y_test_raveled == pred
    print(str(neighbor_itr) + ":" + str(score))
    for idx, tf in enumerate(tf_result):
        if not tf:
            print("Actual:" + str(y_test_raveled[idx]) + "::Guess:" + str(pred[idx]))
            cv.imshow('img', np.reshape(x_test[idx], (-1, 28, 1)).astype(np.uint8))
            if cv.waitKey(0) & 0xFF == ord('q'):
                break


print("finished")
