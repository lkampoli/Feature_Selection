#!/usr/bin/env python
# coding: utf-8

# Feature Selection using Wrappers
# ---
# `scikit learn` does not provide a comprehenisive implementation of Wrapper feature selection so we use `MLxtend`.  
# http://rasbt.github.io/mlxtend/
# So you will probably need to install some libraries:  
# `pip install mlxtend`  
# `pip install joblib`

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from matplotlib.ticker import MaxNLocator

# Forward Sequential Search on segmentation data.
seg_data = pd.read_csv('segmentation-all.csv')
print(seg_data.shape)
print(seg_data.head())

seg_data['Class'].value_counts()

# Data Prep 
# - Extract the data from the dataframe into numpy arrays
# - Split into train and test sets 
# - Apply a [0,1] Scaler. 

y = seg_data.pop('Class').values
X_raw = seg_data.values
X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, random_state=2, test_size=1/2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)
max_k = X_train.shape[1]
X_train.shape, X_test.shape

# Baseline performance evaluation
# Using all features and *k*-NN:  
# - test performance on training data using cross validation,
# - test performance on test data using hold-out. 

kNN = KNeighborsClassifier(n_neighbors=4)
kNN = kNN.fit(X_train,y_train)
y_pred = kNN.predict(X_test)
acc = accuracy_score(y_pred,y_test)
cv_acc = cross_val_score(kNN, X_train, y_train, cv=8)

print("X_Val on training all features: {0:.3f}".format(cv_acc.mean())) 
print("Hold Out testing all features: {0:.3f}".format(acc)) 

# Sequential Forward Selection
# Run SFS with k_features set to (1,max_k) - this will remember the best result.

verb = 0
sfs_forward = SFS(kNN, 
                  k_features= (1, max_k), 
                  forward=True, 
                  floating=False, 
                  verbose=verb,
                  scoring='accuracy',
                  cv=10, n_jobs = -1) # No. of threads depends on the machine.

sfs_forward = sfs_forward.fit(X_train, y_train, 
                              custom_feature_names=seg_data.columns)

# The indexes and names of the features from the best performing subset.
print(sfs_forward.k_feature_idx_)
print(sfs_forward.k_feature_names_)

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

fig1 = plot_sfs(sfs_forward.get_metric_dict(), 
                ylabel='Training Accuracy',
                kind='std_dev')

plt.ylim([0.5, 1])
plt.title('Sequential Forward Selection')
plt.grid()
plt.show()
print(sfs_forward.k_feature_names_)

# Transform the dataset using the selected subset.
X_train_sfs = sfs_forward.transform(X_train)
X_test_sfs = sfs_forward.transform(X_test)

kNN_sfs = kNN.fit(X_train_sfs,y_train)
y_pred = kNN_sfs.predict(X_test_sfs)
acc_SFS = accuracy_score(y_pred,y_test)
cv_acc_SFS = cross_val_score(kNN, X_train_sfs, y_train, cv=8)

print("X_train shape: ", X_train_sfs.shape)
print("X_Val on SFS all features: {0:.3f}".format(cv_acc_SFS.mean())) 
print("Hold Out testing: {0:2d} features selected using SFS: {1:.3f}".format(len(sfs_forward.k_feature_idx_), acc_SFS)) 

# Backward Elimination
# If we set the SFS `forward` parameter to False it performs Backward Elimination.

verb = 1
sfs_backward = SFS(kNN, 
                  k_features=(1, max_k), 
                  forward=False, 
                  floating=False, 
                  verbose=verb,
                  scoring='accuracy',
                  cv=10, n_jobs = -1)

sfs_backward = sfs_backward.fit(X_train, y_train, 
                              custom_feature_names=seg_data.columns)

fig1 = plot_sfs(sfs_backward.get_metric_dict(), 
                ylabel='Accuracy',
                kind='std_dev')

plt.ylim([0.5, 1])
plt.title('Backward Elimination (w. StdDev)')
plt.grid()
plt.show()
print(sfs_backward.k_feature_names_)

sfs_backward.k_feature_idx_, len(sfs_backward.k_feature_idx_)

X_train_be = sfs_backward.transform(X_train)
X_test_be = sfs_backward.transform(X_test)

kNN_be = kNN.fit(X_train_be,y_train)
y_pred = kNN_be.predict(X_test_be)
acc_BE = accuracy_score(y_pred,y_test)
cv_acc_BE = cross_val_score(kNN, X_train_be, y_train, cv=8)

print("X_train shape: ", X_train_be.shape)
print("X_Val on BE all features: {0:.3f}".format(cv_acc_BE.mean())) 
print("Hold Out testing: {0:2d} features selected using BE: {1:.3f}".format(len(sfs_backward.k_feature_idx_), acc_BE)) 

# Plot the overall results
fig, ax = plt.subplots()
width = 0.2

options = ['All', 'SFS', 'BE']
n_feat = [X_train.shape[1], X_train_sfs.shape[1], X_train_be.shape[1]]
accs = [acc,acc_SFS,acc_BE]
xv = [cv_acc.mean(), cv_acc_SFS.mean(), cv_acc_BE.mean()]

y_pos = np.arange(len(options))

p1 = ax.bar(y_pos-width/2, xv, width, align='center', label = 'Train (X-val)',
            color=['blue','blue','blue'],alpha=0.5)

p2 = ax.bar(y_pos+width/2, accs , width, align='center', label = 'Test (Hold-out)',
            color=['g','g','g'],alpha=0.5)

ax.set_ylim([0.7, 1])
ax2 = ax.twinx()

p3 = ax2.plot([0,1,2],n_feat, color = 'red', label = 'Feature Count',
              marker = 'x', ms = 10, linewidth=0)
ax2.set_ylim([0, 20])

ax.grid(axis = 'y')

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='lower right')

ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xticks(y_pos, options)
ax.set_ylabel('Accuracy')
ax2.set_ylabel('Feature Count')

plt.show()
