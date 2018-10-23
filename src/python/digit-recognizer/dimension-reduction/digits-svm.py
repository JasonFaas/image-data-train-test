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
from sklearn.decomposition import PCA


from image_test_space import DisplayImage

resources = '../../../resources/digit-recognizer'
# train_csv = 'train.csv'
# train_csv = 'jason_train_10000.csv'
# train_csv = 'jason_train_5000.csv'
# train_csv = 'jason_train_4000.csv'
train_csv = 'jason_train_2000.csv'
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

# for c_param in [0.001,0.01,0.1,1,10,100]:
from sklearn.svm import SVC
for param_2 in ['auto', 1, .01]:
    for param_1 in ['rbf', 'linear']:
        model_svm = SVC(kernel=param_1, gamma=param_2)
        model_svm.fit(x_train_v2, y_train)
        preds = model_svm.predict(x_test_v2)

        print(len(model_svm.support_))

        x_small = x_train_v2[model_svm.support_]
        y_small = y_train[model_svm.support_]
        model_svm_small = SVC(kernel=param_1, gamma=param_2)
        model_svm_small.fit(x_small, y_small)

        if review_failures:
            guess_vs_actual = preds == y_test
            for idx, good_guess in enumerate(guess_vs_actual):
                if not good_guess:
                    print("Guess " + str(preds[idx]) + " \tActual " + str(y_test[idx]))
                    image = x_test[idx]
                    image = np.reshape(image, (-1, 28, 1))
                    image.astype(np.uint8)
                    print(image.shape)
                    cv.imshow("failure", image)
                    if cv.waitKey(0) & 0xFF == ord('q'):
                        break

        print("Score for gamma of " + str(param_2) + " and kernel of " + param_1 + " " + str(round(model_svm.score(x_test_v2, y_test), 3)))
        print("Score for gamma of " + str(param_2) + " and kernel of \'" + param_1 + "\' " + str(round(model_svm_small.score(x_test_v2, y_test), 3)))
