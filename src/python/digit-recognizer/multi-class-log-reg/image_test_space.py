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
        image_8bit = self.get_image_from_matrix(location)

        cv.imshow("img_2", image_8bit)
        return cv.waitKey(0) & 0xFF
        # cv.destroyAllWindows()

    def get_image_from_matrix(self, location):
        image_values = self.digit_train_set.loc[location, ['image']].values
        image_8bit = self.get_image_from_values(image_values[0])
        return image_8bit

    def get_image_from_values(self, image_values):
        image_1d_array = np.array(image_values)
        image_2d_array = np.reshape(image_1d_array, (-1, 28, 1))
        image_8bit = image_2d_array.astype(np.uint8)
        return image_8bit

    def get_all_info(self):
        new_array = np.array(self.digit_train_set[['label', 'image']].values)
        get_all_images_from_values = list(map(lambda x: self.get_image_from_values(x[1]), new_array))
        new_array[:,1] = get_all_images_from_values
        return new_array


    def rotate_to_upright(self, img):
        # distance between 2 points lambda
        two_point_distance = lambda x, y: int(((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5)

        # contours
        _, contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return img
        cnt = contours[0]

        # vertices of minimum area rectangle
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        # identify top left point
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

        # calculate source points
        src_pts = np.array([box[(top_left_pos + 0) % 4],
                            box[(top_left_pos + 1) % 4],
                            box[(top_left_pos + 2) % 4],
                            box[(top_left_pos + 3) % 4]], dtype="float32")

        # calculate destination points
        top_rect_distance = two_point_distance(box[(top_left_pos + 0) % 4], box[(top_left_pos + 1) % 4])
        left_rect_distance = two_point_distance(box[(top_left_pos + 1) % 4], box[(top_left_pos + 2) % 4])
        if top_rect_distance <= 1 or left_rect_distance <= 1:
            # don't warp if rectangle is too small
            warped_img = img
        else:
            x_start = int((img.shape[1] - top_rect_distance) / 2)
            y_start = int((img.shape[1] - left_rect_distance) / 2)
            dst_pts = np.array([
                [x_start, y_start],
                [x_start + top_rect_distance, y_start],
                [x_start + top_rect_distance, y_start + left_rect_distance],
                [x_start, y_start + left_rect_distance]], dtype="float32")

            # warp image
            matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
            warped_img = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
        return warped_img

    def circle_count_and_locations(self, img):
        ff_mean = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, -2)
        cv.floodFill(ff_mean, np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8), (0, 0), 255)

        ff_mean_inv = cv.bitwise_not(ff_mean)

        cnts = cv.findContours(ff_mean_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]

        top_point = np.zeros((2), np.uint8)
        bottom_point = np.zeros((2), np.uint8)

        for c in cnts:
            M = cv.moments(c)
            if M["m00"] > .1:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                halfway_c = int(c.shape[0] - 1 / 2)
                cX = c[halfway_c, 0, 0]
                cY = c[halfway_c, 0, 1]

            if bottom_point[1] == 0 or cY < bottom_point[1]:
                if top_point[1] == 0:
                    top_point[0] == bottom_point[0]
                    top_point[1] == bottom_point[1]

                bottom_point[0] = cX
                bottom_point[1] = cY
            elif top_point[1] == 0 or cY > top_point[1]:
                top_point[0] == cX
                top_point[1] == cY



        return [len(cnts), bottom_point[0], bottom_point[1], top_point[0], top_point[1]]


    def circle_info_arr(self, samples_1d, samples_2d):
        samples_after = []

        for idx, sample in enumerate(samples_1d):
            circle_info_size = 5
            new_sample = np.zeros((sample.shape[0] + circle_info_size), np.uint8)
            new_sample[0:sample.shape[0]] = sample[:]

            # include circle count
            circle_info = self.circle_count_and_locations(samples_2d[idx])
            for i in range(circle_info_size):
                new_sample[-1 - i] = circle_info[i]
            samples_after.append(new_sample)
        return samples_after
