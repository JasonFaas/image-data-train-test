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

model_pca = PCA(0.90)
x_train_v3 = model_pca.fit_transform(x_train_v2)
x_test_v3 = model_pca.transform(x_test_v2)


review_failures = True

# for c_param in [0.001,0.01,0.1,1,10,100]:
for c_param in [10]:
    for penalty_param in ['l2']:
        logreg = LogisticRegression(penalty=penalty_param, C=c_param, solver='lbfgs')
        logreg.fit(x_train_v3, y_train)
        preds = logreg.predict(x_test_v3)
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

        print("Score for C of " + str(c_param) + " and penalty of " + penalty_param + " " + str(round(logreg.score(x_test_v3, y_test), 3)))
