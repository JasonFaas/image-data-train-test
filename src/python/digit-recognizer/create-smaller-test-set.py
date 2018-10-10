import numpy as np
import cv2 as cv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

resources_path = '../../resources/'
csv_read = '%sdigit-recognizer/train.csv' % resources_path
digit_train_set = pd.read_csv(csv_read)
dts_by_label = {}

total_new_set_size = 1000*4
row_count = str(total_new_set_size)
csv_write = '%sdigit-recognizer/jason_train_%s.csv' % (resources_path, row_count)
label_count = 10


for label_val in range(0, 10):
    save_count = int(total_new_set_size / label_count)
    dts_by_label[label_val] = digit_train_set.loc[digit_train_set['label'] == label_val]
    # print(dts_by_label[label_val].head(save_count))

dts_output = dts_by_label[0].head(save_count)
for label_val in range(1, 10):
    dts_output = pd.concat([dts_output, dts_by_label[label_val].head(save_count)])

dts_output.to_csv(csv_write, index=False)
