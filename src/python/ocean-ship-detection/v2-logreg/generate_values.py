import numpy as np
import cv2 as cv
import pandas as pd


class GenerateValues:

    def __init__(self, image_size, top_bucket_size, second_bucket_sz, training, review_warnings):
        self.review_warnings = review_warnings
        self.training = training
        self.black_images_logged_top = 0
        self.black_images_logged_second = 0
        self.pixels = image_size ** 2
        self.image_sz = image_size
        self.top_bucket_sz = top_bucket_size
        if top_bucket_size == 8:
            self.top_bucket_count = 96
            self.second_bucket_count = 1
            self.second_bucket_sz = second_bucket_sz
        else:
            print("failure...")
            exit(1)

        assert self.image_sz % self.top_bucket_count == 0
        assert int(self.image_sz / self.top_bucket_count) == self.top_bucket_sz

        assert self.image_sz % (self.top_bucket_sz * self.second_bucket_sz) == 0


    def get_info_to_log(self, image_to_log, ship_mask, examine_mask):
        image_sz = image_to_log.shape[0]
        pixels = image_sz ** 2

        y_train = np.count_nonzero(ship_mask) > pixels * .1

        # logging values
        blue_avg = int(round(np.average(image_to_log[:, :, 0])))
        green_avg = int(round(np.average(image_to_log[:, :, 1])))
        red_avg = int(round(np.average(image_to_log[:, :, 2])))
        blue_std = int(round(np.std(image_to_log[:, :, 0])))
        green_std = int(round(np.std(image_to_log[:, :, 1])))
        red_std = int(round(np.std(image_to_log[:, :, 2])))
        values = (blue_avg, green_avg, red_avg, blue_std, green_std, red_std)

        return y_train, values


    def parsing_values(self, image_to_log, ship_mask, examine_mask):
        values_dict = {}

        # examine top_bucket_sz chunks
        for x_top_start in range(0, self.image_sz, self.top_bucket_sz):
            for y_top_start in range(0, self.image_sz, self.top_bucket_sz):
                x_top_stop = x_top_start + self.top_bucket_sz
                y_top_stop = y_top_start + self.top_bucket_sz
                itl_slice = image_to_log[x_top_start:x_top_stop, y_top_start:y_top_stop]
                ship_slice = ship_mask[x_top_start:x_top_stop, y_top_start:y_top_stop]
                examine_slice = examine_mask[x_top_start:x_top_stop, y_top_start:y_top_stop]

                if np.count_nonzero(examine_slice) == 0:
                    continue

                y_train_top, log_values = self.get_info_to_log(itl_slice, ship_slice, examine_slice)

                dict_position_top = str(x_top_start) + "_" + str(y_top_start)
                values_dict[dict_position_top] = [y_train_top, log_values[0], log_values[1], log_values[2], log_values[3], log_values[4], log_values[5]]

        return values_dict

    def parse_second_level_values(self, image_to_log, thresh_mask, training_mask, x_top_start, x_top_stop, y_top_start, y_top_stop):
        train_dict_second_level = {}
        for x_second_start in range(x_top_start, x_top_stop, self.second_bucket_sz):
            for y_second_start in range(y_top_start, y_top_stop, self.second_bucket_sz):
                x_second_stop = x_second_start + self.second_bucket_sz
                y_second_stop = y_second_start + self.second_bucket_sz
                itl_slice = image_to_log[x_second_start:x_second_stop, y_second_start:y_second_stop]
                tm_slice = training_mask[x_second_start:x_second_stop, y_second_start:y_second_stop]
                thresh_slice = thresh_mask[x_second_start:x_second_stop, y_second_start:y_second_stop]
                y_train_second, log_values = self.get_info_to_log(itl_slice, tm_slice, thresh_slice)

                if type(log_values) == type(None):
                    print("fix second level")
                    continue

                if self.training and log_values[0] == -1 and self.black_images_logged_second > 100:
                    continue
                elif log_values[0] == -1:
                    log_values = (0, 0, 0, 0, 0, 0)
                    self.black_images_logged_second += 1

                dict_position_second = str(x_second_start) + "_" + str(y_second_start)
                train_dict_second_level[dict_position_second] = [y_train_second, log_values[0], log_values[1],
                                                                 log_values[2], log_values[3], log_values[4],
                                                                 log_values[5]]
        return train_dict_second_level
