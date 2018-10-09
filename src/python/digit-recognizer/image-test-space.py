import numpy as np
import cv2 as cv
import pandas as pd

# digit_train_set = pd.read_csv('../../resources/digit-recognizer/train.csv')
digit_train_set = pd.read_csv('../../resources/digit-recognizer/jason_train_1000.csv')

independent_columns = digit_train_set.columns[1:]
dependent_column = digit_train_set.columns[0:1]

digit_train_set['image'] = digit_train_set.loc[:, independent_columns].values.tolist()

image_ = digit_train_set.loc[888, ['image']].values
print(type(image_))
out_img = np.array(image_[0])
print(type(out_img))
print(out_img.shape)
out_2 = np.reshape(out_img, (-1, 28, 1))
print(out_2.shape)
out_2 = out_2.astype(np.uint8)

cv.imshow("img_2", out_2)
cv.waitKey(0)
cv.destroyAllWindows()
