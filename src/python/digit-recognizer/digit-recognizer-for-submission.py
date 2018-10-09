import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

resources_path = '../../resources/digit-recognizer/'
# digit_train_set = pd.read_csv('%strain.csv' % resources_path)
# digit_train_set = pd.read_csv('%sjason_train_1000.csv' % resources_path)
digit_train_set = pd.read_csv('%sjason_train_10000.csv' % resources_path)

digit_test_set = pd.read_csv('%stest.csv' % resources_path)
# digit_test_set = pd.read_csv('%sjason_test_1000.csv' % resources_path)


independent_columns = digit_train_set.columns[1:]
dependent_column = digit_train_set.columns[0:1]

x_train = digit_train_set.loc[:, independent_columns].values
y_train = digit_train_set.loc[:, dependent_column].values
x_test = digit_test_set.values

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train.ravel())
pred = knn.predict(x_test)
result_df = pd.DataFrame(pred)
result_df.index += 1
result_df = result_df.reset_index()
result_df.columns = ['ImageId', 'Label']
result_df.to_csv('%sjason_submission.csv' % resources_path, index=False)
