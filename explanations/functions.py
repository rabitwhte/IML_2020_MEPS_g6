"""

Module with functions that can be helpful
for this project

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



def transform_target_log3(data, target_name):
    val = data[target_name].values 
    return np.array([0 if v == 0 else np.log(v) / np.log(3) for v in val])



def read_x_y(path, target_name):
    data = pd.read_csv(path)
    data[target_name] = transform_target_log3(data, target_name)
    X = data.drop(columns = target_name)
    y = data[target_name]
    return X, y



def get_categorical_features(data):
    num_unique = data.nunique()
    categorical_features = num_unique[num_unique <= 10].index.tolist()
    # Remove variables from categorical features list that can be treated as continuous
    for col in ["POVCAT15", "RTHLTH31", "MNHLTH31"]:
        categorical_features.remove(col)
    return categorical_features



def one_hot_scikit(X, not_cat_cols_list):
    categorical_features = get_categorical_features(X, not_cat_cols_list)
    encoder = OneHotEncoder(sparse = False, handle_unknown = "ignore")
    X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
    X_encoded.columns = encoder.get_feature_names(categorical_features)
    X.drop(categorical_features, axis = 1, inplace = True)
    X = pd.concat([X, X_encoded], axis = 1)
    return X



def one_hot(data, not_cat_cols_list):
    categorical_features = get_categorical_features(data, not_cat_cols_list)
    for col in categorical_features:
        df_cat_vars_one_hot = pd.get_dummies(data[col], prefix = col)
        data = data.drop(columns = [col])
        data = pd.concat([data, df_cat_vars_one_hot], axis = 1)
    return data



def print_model_results(y_train, y_pred_train, y_test, y_pred_test):
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    
    print("Training: \nRMSE: ", rmse_train, " | MAE: ", mean_absolute_error(y_train, y_pred_train), " | R^2: ", r2_score(y_train, y_pred_train), "\n")
    print("Test: \nRMSE: ", rmse_test, " | MAE: ", mean_absolute_error(y_test, y_pred_test), " | R^2: ", r2_score(y_test, y_pred_test), "\n")
    

    
def plot_target_pred(target_name, target, pred_name, pred, path):
    plt.figure(figsize = (10, 8))
    plt.scatter(target, pred, s = 90, alpha = 0.7)
    plt.plot(target, target, c = "y", lw = 5)
    plt.xlabel(target_name, size = 14)
    plt.ylabel(pred_name, size = 14)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.savefig(path + target_name[2:] + ".svg")
    plt.show()



def find_nth_obs_idx(y, y_pred, n):
    diff = np.abs(y.values - y_pred)
    diff_args_srt = np.argsort(diff)
    return diff_args_srt[n]



def plot_pvi_boxplots(pvi, n, data, model_name, img_path):
    pvi_sorted_idx = pvi.importances_mean.argsort()
    plt.figure(figsize = (10, 8))
    title = "Permutational variable importance for \n" + model_name
    plt.title(title, size = 14, fontweight = "bold")
    plt.grid()
    plt.boxplot(pvi.importances[pvi_sorted_idx][-n:].T, vert = False, labels = data.columns[pvi_sorted_idx][-n:])
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.savefig(img_path + ".png")
    plt.show()