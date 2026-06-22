import sys
import joblib
import time
from shutil import copy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter

from glob import glob

import math

basis_set = os.getenv("BASIS")
structure = os.getenv("STRUCTURE")
num_files = int(os.getenv("NUM_FILES"))

properties=['Evir1', 'Hvir1', 'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2', 'Eocc1', 'Jocc1', 'Kocc1', 'Hocc1','Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Jia1', 'Jia2', 'Kia1', 'Kia2','diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 'Jia1mag', 'Jia2mag','Kia1mag', 'Kia2mag','t2']

# Open the file in binary read mode and load the data
for i in range(2, num_files + 1):
    file_path = f"Data/{basis_set}/{structure}/pickle/{structure.lower()}{i}_alldata.pkl.gz"

    if i == 2:
        data = pd.read_pickle(file_path)
        num_t2 = len(data['t2'])
        print(num_t2)
    else:
        new_data = pd.read_pickle(file_path)
        data = pd.concat([data, new_data], ignore_index=True)
        print(len(new_data['t2']))

# Data process 
train,test=train_test_split(range(math.ceil(len(data)/num_t2)),train_size=0.8,test_size=0.2)


train_name = [f'{structure}{i}' for i in train]
test_name = [f'{structure}{i}' for i in test]

print(train_name)

X_train=[]
y_train=[]
X_test=[]
y_test=[]

for i in train:
    X_train += [data[i*num_t2: (i + 1) * num_t2].drop(columns=['t2'])]
    y_train += data[i*num_t2: (i + 1) * num_t2]['t2'].to_list()

for i in test:
    X_test += [data[i*num_t2:(i + 1) * num_t2].drop(columns=['t2'])]
    y_test += data[i*num_t2:(i + 1) * num_t2]['t2'].to_list()

X_train = np.array(X_train, dtype=np.float128)
y_train = np.array(y_train, dtype=np.float128)

X_test = np.array(X_test, dtype=np.float128)
y_test = np.array(y_test, dtype=np.float128)

X_train = X_train.reshape(-1, X_train.shape[2])
X_test = X_test.reshape(-1, X_test.shape[2])

scaler = MinMaxScaler
x_scaler = scaler((-1, 1))
y_scaler = scaler((-1, 1))

X_train=x_scaler.fit_transform(X_train)
y_train=y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()

X_test = x_scaler.fit_transform(X_test)
y_test = y_scaler.transform(y_test.reshape(-1,1)).flatten()

print(X_train.shape,y_train.shape)

top5=np.array([20, 21, 24, 23, 22][::-1])
top10=np.array([28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
top15=np.array([18, 5, 9, 29, 27, 28, 8, 13, 26, 12, 20, 21, 24, 23, 22][::-1])
top5_names=np.array(properties)[top5]
top10_names=np.array(properties)[top10]
top15_names=np.array(properties)[top15]

print(X_test.shape)

print(top5_names, top5)
print(top10_names, top10)
print(top15_names, top15)
print(properties)

print(X_train[10][top5])

scaled_features = f'./Blind_Data/{structure}/scaled/'
x_scale_file = scaled_features + "feature_scaler.bin"
y_scale_file = scaled_features + "t2_scaler.bin"

with open(y_scale_file, "wb") as f:
    joblib.dump(y_scaler, f)
    f.close()
with open(x_scale_file, "wb") as f:
    joblib.dump(x_scaler, f)
    f.close()

#stores the top 5 features
with open(scaled_features + "5_feature_train.bin", "wb") as f:
    joblib.dump({"X":X_train[:,top5], "y":y_train, "names":train_name, "features":top5_names}, f)
    f.close()
with open(scaled_features + "5_feature_test.bin", "wb") as f:
    joblib.dump({"X":X_test[:,top5], "y":y_test, "names":test_name, "features":top5_names}, f)
    f.close()

#stores the top 10 features
with open(scaled_features + "10_feature_train.bin", "wb") as f:
    joblib.dump({"X":X_train[:,top10], "y":y_train,"names":train_name, "features":top10_names}, f)
    f.close()
with open(scaled_features + "10_feature_test.bin", "wb") as f:
    joblib.dump({"X":X_test[:,top10], "y":y_test, "names":test_name, "features":top10_names}, f)
    f.close()

#stores the top 15 features
with open(scaled_features + "15_feature_train.bin", "wb") as f:
    joblib.dump({"X":X_train[:,top15], "y":y_train, "names":train_name, "features":top15_names}, f)
    f.close()
with open(scaled_features + "15_feature_test.bin", "wb") as f:
    joblib.dump({"X":X_test[:,top15], "y":y_test, "names":test_name, "features":top15_names}, f)
    f.close()

#stores all features
with open(scaled_features + "full_feature_train.bin", "wb") as f:
    joblib.dump({"X":X_train, "y":y_train, "names":train_name, "features":properties}, f)
    f.close()
with open(scaled_features + "full_feature_test.bin", "wb") as f:
    joblib.dump({"X":X_test, "y":y_test, "names":test_name, "features":properties}, f)
    f.close()
