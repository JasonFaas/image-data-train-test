import numpy as np
import cv2 as cv
import pandas as pd


class GenerateValues:

    def __init__(self, image_size, top_bucket_size, second_bucket_sz, training):
        self.training = training
        self.black_images_logged_top = 0
        self.black_images_logged_second = 0
        self.pixels = image_size ** 2
        self.image_sz = image_size
        if top_bucket_size == 48:
            self.top_bucket_sz = 48
            self.top_bucket_count = 16
            self.second_bucket_count = 12
            self.second_bucket_sz = 4
        elif top_bucket_size == 32:
            self.top_bucket_sz = 32
            self.top_bucket_count = 24
            self.second_bucket_count = 8
            self.second_bucket_sz = 4
        elif top_bucket_size == 16:
            self.top_bucket_sz = 16
            self.top_bucket_count = 48
            self.second_bucket_count = 4
            self.second_bucket_sz = 4
        else:
            print("failure...")
            exit(1)

        assert self.image_sz % self.top_bucket_count == 0
        assert int(self.image_sz / self.top_bucket_count) == self.top_bucket_sz

        assert self.top_bucket_sz % self.second_bucket_count == 0
        assert int(self.top_bucket_sz / self.second_bucket_count) == self.second_bucket_sz


    def get_info_to_log(self, image_to_log, training_mask, thresh_mask):
        image_sz = image_to_log.shape[0]
        pixels = image_sz ** 2
        y_train = np.count_nonzero(training_mask) > pixels * .1
        # TODO make this more efficient
        thresh_approval = pixels - np.count_nonzero(thresh_mask) > pixels * .1

        # logging values
        blue_avg = int(round(np.average(image_to_log[:, :, 0])))
        green_avg = int(round(np.average(image_to_log[:, :, 1])))
        red_avg = int(round(np.average(image_to_log[:, :, 2])))
        blue_std = int(round(np.std(image_to_log[:, :, 0])))
        green_std = int(round(np.std(image_to_log[:, :, 1])))
        red_std = int(round(np.std(image_to_log[:, :, 2])))
        values = (blue_avg, green_avg, red_avg, blue_std, green_std, red_std)

        if y_train and not thresh_approval:
            print("Blurring out valid image!\t" + str(image_sz))

            cv.imshow("itl_slice", image_to_log)
            cv.imshow("training_mask_slice", training_mask)
            thresh_mask = thresh_mask.astype(dtype=np.uint8)
            thresh_mask[:, :] *= 200
            cv.imshow("thresh_mask_slice", thresh_mask)

            if image_sz > 10 and cv.waitKey(10) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                exit(0)
            # values = None
        elif not y_train and not thresh_approval:
            values = (-1, -1, -1, -1, -1, -1)

        return y_train, values


    def parsing_values(self, image_to_log, training_mask, thresh_mask):
        train_dict_top_level = {}
        train_dict_second_level = {}

        for x_top_start in range(0, self.image_sz, self.top_bucket_sz):
            for y_top_start in range(0, self.image_sz, self.top_bucket_sz):
                x_top_stop = x_top_start + self.top_bucket_sz
                y_top_stop = y_top_start + self.top_bucket_sz
                itl_slice = image_to_log[x_top_start:x_top_stop, y_top_start:y_top_stop]
                tm_slice = training_mask[x_top_start:x_top_stop, y_top_start:y_top_stop]
                thresh_slice = thresh_mask[x_top_start:x_top_stop, y_top_start:y_top_stop]
                y_train_top, log_values = self.get_info_to_log(itl_slice, tm_slice, thresh_slice)
                # TODO fix warning
                if type(log_values) == type(None):
                    print(str(x_top_start) + " :: " + str(y_top_start))
                    thresh_mask_dsp = thresh_mask.astype(dtype=np.uint8)
                    thresh_mask_dsp[:,:] *= 200

                    cv.imshow("image_to_log", image_to_log)
                    cv.imshow("training_mask", training_mask)
                    cv.imshow("thresh_mask", thresh_mask_dsp)
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        cv.destroyAllWindows()
                        exit(0)
                    else:
                        print("Put more logging in here")
                        continue
                # skip adding info if effectively a black image
                if self.training and log_values[0] == -1 and self.black_images_logged_top > 100:
                    continue
                elif log_values[0] == -1:
                    log_values = (0, 0, 0, 0, 0, 0)
                    self.black_images_logged_top += 1
                dict_position_top = str(x_top_start) + "_" + str(y_top_start)
                train_dict_top_level[dict_position_top] = [y_train_top, log_values[0], log_values[1], log_values[2], log_values[3], log_values[4], log_values[5]]

                # Second level
                if self.training and y_train_top:
                    train_dict_second_level = self.parse_second_level_values(image_to_log,
                                                                             thresh_mask,
                                                                             training_mask,
                                                                             x_top_start,
                                                                             x_top_stop,
                                                                             y_top_start,
                                                                             y_top_stop)
        return train_dict_top_level, train_dict_second_level

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
