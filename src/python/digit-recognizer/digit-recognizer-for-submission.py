import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

resources_path = '../../resources/digit-recognizer/'
# digit_train_set = pd.read_csv('%strain.csv' % resources_path)
digit_train_set = pd.read_csv('%sjason_train_1000.csv' % resources_path)
# digit_train_set = pd.read_csv('%sjason_train_10000.csv' % resources_path)

digit_test_set = pd.read_csv('%stest.csv' % resources_path)
# digit_test_set = pd.read_csv('%sjason_test_1000.csv' % resources_path)


independent_columns = digit_train_set.columns[1:]
dependent_column = digit_train_set.columns[0:1]

x_train = digit_train_set.loc[:, independent_columns].values
y_train = digit_train_set.loc[:, dependent_column].values
x_test = digit_test_set.values

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train.ravel())
print(x_test.shape)
yet_to_process = x_test.shape[0]
processed = 0
max_to_test = 100
result_df = pd.DataFrame(columns=['ImageId', 'Label'])
progress = 0
while yet_to_process > 0:
    # identify how many to process this round
    to_test = min(max_to_test, yet_to_process)
    x_test_sub = x_test[processed:to_test + processed]

    # process test data
    pred = knn.predict(x_test_sub)

    # put data into dataframe
    temp_df = pd.DataFrame(pred)
    temp_df.index += (1 + processed)
    temp_df = temp_df.reset_index()
    temp_df.columns = ['ImageId', 'Label']

    # merge data
    result_df = pd.concat([result_df, temp_df])

    # update processing info
    yet_to_process -= to_test
    processed += to_test

    # progress
    new_progress = int(100 * processed / x_test.shape[0])
    if new_progress != progress:
        progress = new_progress
        print(str(processed) + ":" + str(progress))


result_df.to_csv('%sjason_submission.csv' % resources_path, index=False)
