import pandas as pd
import numpy as np
import random
import sklearn
import tools
from sklearn.compose import make_column_selector as selector
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import warnings
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT

warnings.simplefilter('ignore')


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=",")
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
              'label']
print(df.shape)
# dropping noising features
df = df.drop(columns=['fnlwgt', 'education'])

# replacing all invalid entries by NaN
df = tools.clean_df(df)

print(df.shape)


labels = df.loc[:, 'label'].to_numpy()  # array ot label string
df = df.iloc[:, 0:-1]  # drop label in dataframe
# do label encoding on Y
le = preprocessing.LabelEncoder()
le.fit(labels)
# print("classes = ", le.classes_)
Y = le.transform(labels)

# select the columns with discrete variables
discrete_columns_selector = selector(dtype_include=object)
discrete_columns = discrete_columns_selector(df)
df_discrete = df[discrete_columns]

# replacing nan values with mode of respective column
for (column_name, column_data) in df_discrete.iteritems():
    mode = df_discrete[column_name].mode(dropna=True)[0]
    df_discrete[column_name] = df_discrete[column_name].fillna(mode)

# select the columns with continuous variables
continuous_columns_selector = selector(dtype_include=int)
continuous_columns = continuous_columns_selector(df)
df_continuous = df[continuous_columns]
df_continuous.reset_index(drop=True, inplace=True)

# replacing nan values with mean of respective column
for (column_name, column_data) in df_continuous.iteritems():
    mean = df_continuous[column_name].mean(skipna=True)
    df_continuous[column_name] = df_continuous[column_name].fillna(mean)


# do one hot encoding on dsicrete variables in dataframe
discrete_encoded = tools.oneHotEncoding(df_discrete)
discrete_encoded.reset_index(drop=True, inplace=True)

# merge the continuous and discrete and label features into one df
df_encoded = pd.concat([df_continuous, discrete_encoded], axis=1, join='inner')

X = df_encoded.to_numpy()
print("shape = ", df_encoded.shape)

# KNN_cross_val = tools.l_fold_cross_validation_KNN(L=5, X=X, Y=Y, K=[k for k in range(1, 21)])
# argmax = KNN_cross_val.argmax()
# print("best accuracy = ", KNN_cross_val.max(), ", acheived for K = ", argmax + 1)


# DT_cross_val = tools.l_fold_cross_validation_DT(L=5, X=X, Y=Y, criterion=["gini", "entropy"],
#                                                 splitter=["best", "random"], max_depth=[10, 20, 30, 40, 50],
#                                                 min_impurity_decrease=[0.2, 0.1, 0.05, 0.025, 0.01, 0],
#                                                 min_samples_leaf=[1, 5, 10, 15, 20, 25])
#
# best_hp = DT_cross_val[DT_cross_val.accuracy == DT_cross_val.accuracy.max()]
# # if we have a tie of best hyperparameters, choose the one with the fastest time
# if best_hp.shape[0] > 1:
#     best_hp = best_hp[best_hp.time_req == best_hp.time_req.min()]
# print("best hyperparmater values are: ", best_hp)

# KNN
# for l in [10, 15, 20, 25]:
#     print("l = ", l)
#     KNN_cross_val = tools.l_fold_cross_validation_KNN(L=l, X=X, Y=Y, K=[5, 10, 14, 15])
#     argmax = KNN_cross_val.argmax()
#     print("the best accuracy = ", KNN_cross_val.max(), ", acheived for K = ", argmax + 1)
#
# # decision tree
# for l in [10, 15, 20, 25]:
#     print("l = ", l)
#     DT_cross_val = tools.l_fold_cross_validation_DT(L=l, X=X, Y=Y, criterion=["gini", "entropy"],
#                                                     splitter=["best", "random"], max_depth=[10, 15],
#                                                     min_impurity_decrease=[0.1, 0.025, 0],
#                                                     min_samples_leaf=[1, 5, 10, 15, 20, 25])
#     best_hp = DT_cross_val[DT_cross_val.accuracy == DT_cross_val.accuracy.max()]
#     # if we have a tie of best hyperparameters, choose the one with the fastest time
#     if best_hp.shape[0] > 1:
#         best_hp = best_hp[best_hp.time_req == best_hp.time_req.min()]
#     print("the best hyperparmater values are:", best_hp)

# load and clean test data

df_test = pd.read_csv("C:/Users/admin/Desktop/COMP 551/Assignments/COMP551_A1/adult.test", sep=",")
df_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                   'label']
df_test = df_test.drop(columns=['fnlwgt', 'education'])

# replacing all invalid entries by NaN
df_test = tools.clean_df(df_test)
labels = df_test.loc[:, 'label'].to_numpy()  # array ot label string
df_test = df_test.iloc[:, 0:-1]  # drop label in dataframe
# do label encoding on Y
le = preprocessing.LabelEncoder()
le.fit(labels)
# print("classes = ", le.classes_)
Y_test = le.transform(labels)

# select the columns with discrete variables
discrete_columns_selector = selector(dtype_include=object)
discrete_columns = discrete_columns_selector(df_test)
df_discrete = df_test[discrete_columns]

# replacing nan values with mode of respective column
for (column_name, column_data) in df_discrete.iteritems():
    mode = df_discrete[column_name].mode(dropna=True)[0]
    df_discrete[column_name] = df_discrete[column_name].fillna(mode)

# select the columns with continuous variables
continuous_columns_selector = selector(dtype_include=int)
continuous_columns = continuous_columns_selector(df_test)
df_continuous = df_test[continuous_columns]
df_continuous.reset_index(drop=True, inplace=True)

# replacing nan values with mean of respective column
for (column_name, column_data) in df_continuous.iteritems():
    mean = df_continuous[column_name].mean(skipna=True)
    df_continuous[column_name] = df_continuous[column_name].fillna(mean)


# do one hot encoding on dsicrete variables in dataframe
discrete_encoded = tools.oneHotEncoding(df_discrete)
discrete_encoded.reset_index(drop=True, inplace=True)

# merge the continuous and discrete and label features into one df
df_encoded_test = pd.concat([df_continuous, discrete_encoded,pd.DataFrame(columns=["N"])], axis=1, join='inner')

X_test = df_encoded_test.to_numpy()
print(df_encoded.columns.dtype, df_encoded_test.columns)

print(X_test)
# KNN
for k in [2, 14]:
    # see training data results on KNN using K=k
    KNN_classifier = KNN(n_neighbors=k)
    KNN_classifier.fit(X, Y)  # fit the model using training data
    Y_prediction = KNN_classifier.predict(X_test)  # predict test data
    accuracy = Y_prediction == Y_test
    accuracy = np.sum(accuracy)  # number of correct predictions
    score = accuracy / X_test.shape[0]  # ratio of correct prediction
    print("for k = ", k, "the accuracy on the test data is: ", score)

# DT
# predict on test data using best hyperparameters
DT_classifier = DT(criterion=best_hp.iloc[0, 0], splitter=best_hp.iloc[0, 1], max_depth=best_hp.iloc[0, 2],
                   min_impurity_decrease=best_hp.iloc[0, 3], min_samples_leaf=best_hp.iloc[0, 4])
DT_classifier.fit(X, Y)
Y_prediction = DT_classifier.predict(X_test)
accuracy = Y_prediction == Y_test
accuracy = np.sum(accuracy)  # number of correct predictions
score = accuracy / X_test.shape[0]
print("For the best hyperparameters, the accuracy is: ", score)
sklearn.tree.plot_tree(DT_classifier, max_depth=7)