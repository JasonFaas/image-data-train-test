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
target = target.astype(int)

# print(type(target[0]))
# print(target)
# exit(0)

samples_v2 = list(map(lambda v: np.reshape(v, (-1)), samples_v1))
samples_v3 = image_info.circle_info_arr(samples_v2, samples_v1)




x_train, x_test_before, y_train, y_test = train_test_split(samples_v3,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test_before)

from sklearn.decomposition import PCA
model_pca = PCA(0.80)
x_train = model_pca.fit_transform(x_train)
x_test = model_pca.transform(x_test)

review_failures = True

for c_param in [1]:
    for penalty in ['l1']:
        print("Starting LogReg")
        clf = LogisticRegression(penalty=penalty, C=c_param)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)

        print("\nC " + str(c_param))
        print("P " + str(penalty))
        print(round(clf.score(x_test, y_test), 3))

        if review_failures:
            guess_vs_actual = preds == y_test
            for idx, good_guess in enumerate(guess_vs_actual):
                if not good_guess:
                    print("Guess " + str(preds[idx]) + " \tActual " + str(y_test[idx]))
                    image = x_test_before[idx]
                    print(image[28*28:])
                    image = np.reshape(image[0:28*28], (-1, 28, 1))
                    image.astype(np.uint8)
                    print(image.shape)
                    cv.imshow("failure", image)
                    if cv.waitKey(0) & 0xFF == ord('q'):
                        break


