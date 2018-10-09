import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

resources_path = '../../resources/digit-recognizer/'
# digit_train_set = pd.read_csv('%strain.csv' % resources_path)
digit_train_set = pd.read_csv('%sjason_train_1000.csv' % resources_path)
# digit_train_set = pd.read_csv('%sjason_train_10000.csv' % resources_path)

digit_test_set = pd.read_csv('%stest.csv' % resources_path)
# digit_test_set = pd.read_csv('%sjason_test_1000.csv' % resources_path)

# output info
output_filename = '%sjason_submission.csv' % resources_path
output_columns = ['ImageId', 'Label']

# distinguish columns
independent_columns = digit_train_set.columns[1:]
dependent_column = digit_train_set.columns[0:1]

# split train-test data
x_train = digit_train_set.loc[:, independent_columns].values
y_train = digit_train_set.loc[:, dependent_column].values
x_test = digit_test_set.values

# fit training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train.ravel())

# setup variables for loop
yet_to_process = x_test.shape[0]
processed = 0
max_to_test = 100
progress = 0


while yet_to_process > 0:
    # identify how many to process this round
    to_test = min(max_to_test, yet_to_process)
    x_test_sub = x_test[processed:to_test + processed]

    # process test data
    pred = knn.predict(x_test_sub)

    # put data into dataframe
    result_df = pd.DataFrame(pred)
    result_df.index += (1 + processed)
    result_df = result_df.reset_index()
    result_df.columns = output_columns

    # write data to file
    if processed == 0:
        result_df.to_csv(output_filename, index=False)
    else:
        result_df.to_csv('output_filename.csv', mode='a', index=False, header=False)

    # update processing info
    yet_to_process -= to_test
    processed += to_test

    # progress
    new_progress = int(100 * processed / x_test.shape[0])
    if new_progress != progress:
        progress = new_progress
        print(str(progress) + "%")

