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

digit_train_set[:, 1] = list(map(lambda v: cv.dilate(v, np.ones((1,3),dtype=np.uint8)), digit_train_set[:, 1]))
digit_train_set[:, 1] = list(map(lambda v: cv.threshold(v, 150, 255, cv.THRESH_BINARY)[1], digit_train_set[:, 1]))
digit_train_set[:, 1] = list(map(lambda v: cv.erode(v, np.ones((1,3),dtype=np.uint8)), digit_train_set[:, 1]))

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
def show_other_info(failed_img):
    bounded_img = failed_img.copy()
    _, contours, hierachy = cv.findContours(bounded_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(bounded_img, (x,y), (x+w, y+h), (200), 2)
    cv.imshow('bounded_rect', bounded_img)

    rotated_rectangle_img = failed_img.copy()
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(rotated_rectangle_img, [box], 0, (200), 2)
    cv.imshow('rotated', rotated_rectangle_img)

    # rotate box
    rotate_img = failed_img.copy()
    min_y_1 = 21
    min_y_pos_1 = 5
    min_y_2 = 22
    min_y_pos_2 = 6
    for idx, val in enumerate(box):
        current_y = val[1]
        if current_y < min_y_1:
            min_y_2 = min_y_1
            min_y_1 = current_y
            min_y_pos_2 = min_y_pos_1
            min_y_pos_1 = idx
        elif current_y < min_y_2:
            min_y_pos_2 = idx
            min_y_2 = current_y
    if ((min_y_pos_1 + 1) % 4) == min_y_pos_2:
        top_left_pos = min_y_pos_1
    else:
        top_left_pos = (min_y_pos_1 - 1) % 4

    two_point_distance = lambda x, y: int(((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5)

    src_pts = np.array([box[(top_left_pos + 0) % 4],
                        box[(top_left_pos + 1) % 4],
                        box[(top_left_pos + 2) % 4],
                        box[(top_left_pos + 3) % 4]], dtype="float32")
    top_rect_distance = two_point_distance(box[(top_left_pos + 0) % 4], box[(top_left_pos + 1) % 4])
    left_rect_distance = two_point_distance(box[(top_left_pos + 1) % 4], box[(top_left_pos + 2) % 4])
    if top_rect_distance == 0 or left_rect_distance == 0:
        warped_img = rotate_img
    else:
        x_start = int((rotate_img.shape[1] - top_rect_distance) / 2)
        y_start = int((rotate_img.shape[1] - left_rect_distance) / 2)
        dst_pts = np.array([
            [x_start, y_start],
            [x_start + top_rect_distance, y_start],
            [x_start + top_rect_distance, y_start + left_rect_distance],
            [x_start, y_start + left_rect_distance]], dtype="float32")
        matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv.warpPerspective(rotate_img, matrix, (rotate_img.shape[1], rotate_img.shape[0]))

    cv.imshow('warped', warped_img)







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
            failed_img = np.reshape(x_test[idx], (-1, 28, 1)).astype(np.uint8)
            cv.imshow('img', failed_img)
            show_other_info(failed_img)
            if cv.waitKey(0) & 0xFF == ord('q'):
                break


print("finished")
