import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier as DT
import matplotlib.pyplot as plt


def clean_df(df):  # deletes all instances with 3 or more missing values
    ncols = df.shape[1]
    l = []  # list with all indices of rows to delete
    # iterate over all the rows
    for index, row in df.iterrows():
        num_of_na = 0
        for i in range(ncols):
            if row[i] == " ?":
                df.iat[index, i] = np.nan
    return df


def oneHotEncoding(df_discrete):
    # from OneHotEncoder library
    encoder = OneHotEncoder(sparse=False)
    df_encoded = encoder.fit_transform(df_discrete)
    columns_encoded = encoder.get_feature_names(df_discrete.columns)
    return pd.DataFrame(df_encoded, columns=columns_encoded)


def l_fold_cross_validation_KNN(L, X, Y, K):
    total_time_start = time.perf_counter()
    size_k = len(K)
    column_names = ["k=" + str(i) for i in K]
    X_rows, Y_rows = X.shape[0], Y.shape[0]
    assert X_rows == Y_rows, "X and Y are not of same length"
    split_value = int(X_rows / L)
    # print("split_value = ", split_value)

    result = [0] * size_k  # list containing the validation results for specific p
    for l in range(0, L):
        result_index=0
        print(int(l*100/L),"%")
        # validation sets have size = split_value
        X_validation, Y_validation = X[l * split_value:(l + 1) * split_value, :], Y[l * split_value:(l + 1) * split_value]
        # training set is initial set without the validation set
        X_training, Y_training = np.delete(X, list(range(l * split_value, (l + 1) * split_value)),
                                           axis=0), np.delete(Y, list(
            range(l * split_value, (l + 1) * split_value)), axis=0)
        # make sure validation set has length split_value
        assert X_validation.shape[0] == split_value, "Split incorrect"

        for k in K:
            classifier = KNN(n_neighbors=k)
            classifier.fit(X_training, Y_training)
            Y_prediction = classifier.predict(X_validation)
            accuracy = Y_prediction == Y_validation
            accuracy = np.sum(accuracy)  # number of correct predictions
            score = accuracy / split_value  # ratio of correct predictions
            result[result_index] += score
            result_index+=1
    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start
    results = np.divide(result, L)
    return results

def l_fold_cross_validation_KNN_scaling_test(L, X, Y, K, scalable_features):
    total_time_start = time.perf_counter()
    column_names = ["k=" + str(i) for i in range(1, K + 1)]
    X_rows, Y_rows = X.shape[0], Y.shape[0]
    assert X_rows == Y_rows, "X and Y are not of same length"
    split_value = int(X_rows / L)
    
    # store our results in a dict:
    results = {}
    
    # test out different scale values
    scale_values = (10, 25, 50, 100, 150 , 200)
    for feature in scalable_features:
        for scale in scale_values:
            X[feature] = scale * X[feature]
            print('Now scaling ' + feature + ' by ' + str(scale))
            for l in range(0, L):
                # validation sets have size = split_value
                X_data = X.to_numpy()
                X_validation, Y_validation = X_data[l * split_value:(l+1)*split_value, :], Y[l * split_value:(l+1)*split_value]
                # training set is initial set without the validation set
                X_training, Y_training = np.delete(X_data, list(range(l * split_value, (l + 1) * split_value)),
                                                   axis=0), np.delete(Y, list(
                    range(l * split_value, (l + 1) * split_value)), axis=0)
                # make sure validation set has length split_value
                assert X_validation.shape[0] == split_value, "Split incorrect"
        
                classifier = KNN(n_neighbors=K)
                classifier.fit(X_training, Y_training)
                Y_prediction = classifier.predict(X_validation)
                accuracy = Y_prediction == Y_validation
                accuracy = np.sum(accuracy)  # number of correct predictions
                score = accuracy / split_value  # ratio of correct predictions
                if feature not in results:
                    results[feature] = {}
                if scale in results[feature]:
                    results[feature][scale] += score
                else:
                    results[feature][scale] = 0
                    results[feature][scale] += score
                # print("L = ", l + 1, "K = ", K, "score = ", score)
            results[feature][scale] = np.divide(results[feature][scale], L)
            # resetting scale
            X[feature] = X[feature].div(scale)
    return results


def plot_scaling_results(results):
    for feature in results:
        xpoints = []
        ypoints = []
        for scale in results[feature].keys():
            xpoints.append(scale)
            ypoints.append(results[feature][scale])
        plt.plot(xpoints, ypoints, marker='*', label=feature)
    plt.legend(loc=0)
    plt.xlabel("Feature scale")
    plt.ylabel("Accuracy")
    plt.show()

# criterion ={"gini", "entropy"}, splitter = {"best","random"}, max_depth = int, min_impurity_decrease = int in [0,1], min_samples_leaf = int
def l_fold_cross_validation_DT(L, X, Y, criterion=["gini"], splitter=["best"], max_depth=None, min_impurity_decrease=[0.0], min_samples_leaf=[1]):
    total_time_start = time.perf_counter()
    column_names = ["criterion", "splitter", "max_depth", "min_impurity_decrease", "min_samples_leaf", "accuracy","time_req"]
    permutations = len(criterion) * len(splitter) * len(max_depth) * len(min_impurity_decrease) * len(
        min_samples_leaf)
    scores = [0] * permutations  # store score value for each row in results dataframe
    timer = [0]*permutations
    results = pd.DataFrame(columns=column_names)
    X_rows, Y_rows = X.shape[0], Y.shape[0]
    assert X_rows == Y_rows, "X and Y are not of same length"
    split_value = int(X_rows / L)
    
    for l in range(0, L):
        print(int(l*100/L),"%")
        row_index = 0
        # validation sets have size = split_value
        X_validation, Y_validation = X[l * split_value:(l + 1) * split_value, :], Y[l * split_value:(l + 1) * split_value]
        # training set is initial set without the validation set
        X_training, Y_training = np.delete(X, list(range(l * split_value, (l + 1) * split_value)), axis=0), \
                                 np.delete(Y, list(range(l * split_value, (l + 1) * split_value)), axis=0)
        # make sure validation set has length split_value
        assert X_validation.shape[0] == split_value, "Split incorrect"
        
        
        for criteria in criterion:
            for split in splitter:
                for m_d in max_depth:
                    for m_i_d in min_impurity_decrease:
                        for m_s_l in min_samples_leaf:
                            time_start = time.perf_counter()
                            classifier = DT(criterion=criteria, splitter=split, max_depth=m_d,
                                            min_impurity_decrease=m_i_d, min_samples_leaf=m_s_l)
                            classifier.fit(X_training, Y_training)
                            Y_prediction = classifier.predict(X_validation)
                            time_end = time.perf_counter()
                            accuracy = Y_prediction == Y_validation
                            accuracy = np.sum(accuracy)  # number of correct predictions
                            scores[row_index] += accuracy / split_value  # ratio of correct predictions
                            timer[row_index] += time_end-time_start
                            row = [criteria, split, m_d, m_i_d, m_s_l, scores[row_index], timer[row_index]]
                            results.loc[row_index] = row
                            # print("row", row_index, ": ", row)
                            row_index += 1
            
    print("Done")
    total_time_end = time.perf_counter()
    total_time = total_time_end - total_time_start
    print("Performing " + str(L) + "-fold-cross-validation on ", permutations, " permutations of hyperparameters took ", total_time, " seconds")
    results['accuracy'] = results['accuracy'].div(L)
    return results
