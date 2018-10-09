import numpy as np
import cv2 as cv
import pandas as pd

class DisplayImage:

    def __init__(self, file_path=None, data_set=None):
        if type(file_path) != type(None):
            self.digit_train_set = pd.read_csv(file_path)
            independent_columns = self.digit_train_set.columns[1:]
            self.digit_train_set['image'] = self.digit_train_set.loc[:, independent_columns].values.tolist()
        elif type(data_set) != type(None):
            self.digit_train_set = pd.DataFrame({'image':data_set.tolist()})



    def display_image(self, location=None):
        image_values = self.digit_train_set.loc[location, ['image']].values
        image_1d_array = np.array(image_values[0])
        image_2d_array = np.reshape(image_1d_array, (-1, 28, 1))
        image_8bit = image_2d_array.astype(np.uint8)

        cv.imshow("img_2", image_8bit)
        return cv.waitKey(0) & 0xFF
        # cv.destroyAllWindows()


csv = '../../resources/digit-recognizer/jason_train_1000.csv'
# csv = '../../resources/digit-recognizer/train.csv'
# display_image = DisplayImage(file_path=csv)
# display_image.display_image(888)
