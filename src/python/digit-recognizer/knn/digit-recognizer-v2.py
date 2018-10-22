import numpy as np
import cv2 as cv
import operator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from image_test_space import DisplayImage

csv_filename = '../../resources/digit-recognizer/jason_train_2000.csv'
# csv_filename = '../../resources/digit-recognizer/train.csv'
digit_train_set = pd.read_csv(csv_filename)

image_info = DisplayImage(csv_filename)
digit_train_set = image_info.get_all_info()

digit_train_set[:, 1] = list(map(lambda v: cv.dilate(v, np.ones((1,3),dtype=np.uint8)), digit_train_set[:, 1]))
digit_train_set[:, 1] = list(map(lambda v: cv.threshold(v, 100, 255, cv.THRESH_TOZERO)[1], digit_train_set[:, 1]))
digit_train_set[:, 1] = list(map(lambda v: image_info.rotate_to_upright(v), digit_train_set[:, 1]))

x = digit_train_set[:,1]
y = digit_train_set[:,0]
y = y.astype('int')

x = list(map(lambda v: np.reshape(v, (-1)), x))
k_scores = {}

review_images = False

for rand_state in range(80, 85):
    print("random_state:" + str(rand_state))
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=rand_state)

    # creating loop for neighbor_itr for later
    # 3 is best, though 1-10 are very similar at about 93.5%


    for neighbor_itr in range(1, 5):
        print("\tk:" + str(neighbor_itr))
        knn = KNeighborsClassifier(n_neighbors=neighbor_itr)
        knn.fit(x_train, y_train.ravel())
        pred = knn.predict(x_test)
        y_test_raveled = y_test.ravel()
        score = accuracy_score(y_test_raveled, pred)
        tf_result = y_test_raveled == pred
        if not k_scores.get(neighbor_itr):
            k_scores[neighbor_itr] = {}
        k_scores[neighbor_itr][rand_state] = score
        if review_images:
            for idx, tf in enumerate(tf_result):
                if not tf:
                    print("Actual:" + str(y_test_raveled[idx]) + "::Guess:" + str(pred[idx]))
                    failed_img = np.reshape(x_test[idx], (-1, 28, 1)).astype(np.uint8)
                    cv.imshow('img', failed_img)
                    wait_key = cv.waitKey(0) & 0xFF
                    if wait_key == ord('q'):
                        break
                    elif wait_key == 27:
                        exit(0)

min_scores = {}
for i in k_scores.items():
    min_scores[i[0]] = sorted(i[1].items(), key=operator.itemgetter(1))
max_min_score = sorted(min_scores.items(), key=operator.itemgetter(1), reverse=True)
print("best minimum k:" + str(max_min_score[0][0]))
min_score_of_max = sorted(max_min_score[0][1], key=operator.itemgetter(1))
print("worst score of k:" + str(min_score_of_max[0][1]))

print("\nfinished")
