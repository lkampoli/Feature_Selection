#!/usr/bin/env python
# coding: utf-8

# Correlation Based Feature Selection (CFS)
# This code walks through two examples of Correlation Based Feature Selection 
# using the Segmentation and Penguin datasets. Two search techniques are 
# presented here, a Forward search and a Best First search. Some functions 
# used in this notebook are from the CFS implementation by Jundong et al [1].

# Import Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from CFS import cfs, merit_calculation
from CFS_ForwardSearch import CFS_FS

# Example 1: CFS on Segmentation dataset
#seg_data = pd.read_csv('segmentation-all.csv')
#seg_data = pd.read_csv('../../data/STS/transport/boltzmann/shear_viscosity.csv')
seg_data = pd.read_csv('../../data/MT/DB6Tr.csv')
print(seg_data.shape)
print(seg_data.head())

#y = seg_data.pop('Class').values
y = seg_data.pop('Viscosity').values
X_raw = seg_data.values

#X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, random_state=2, test_size=1/2)
X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, train_size=0.75, test_size=0.25, random_state=666, shuffle=False)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)
max_length = X_train.shape[0]
feat_num = X_train.shape[1]
print(X_train.shape, X_test.shape, y_train.shape)

#kNN = KNeighborsClassifier(n_neighbors=5)
kNN = KNeighborsRegressor(n_neighbors=5)

kNN = kNN.fit(X_train, y_train)
y_pred = kNN.predict(X_test)
#acc = accuracy_score(y_pred, y_test)
acc = r2_score(y_pred, y_test)
cv_acc = cross_val_score(kNN, X_train, y_train, cv=5)

print("X_Val on training all features: {0:.3f}".format(cv_acc.mean())) 
print("Hold Out testing all features: {0:.3f}".format(acc)) 

# Forward Search - CFS
# Here, the best feature subset is found by finding the best single feature 
# subset using merit score then adding all other features to this best feature 
# and recomputing the merit. This process continues until merit score stops increasing.

merit_score_sel, sel_comb = CFS_FS(X_train, y_train)
print("Merit Score of Selected Features: " + str(merit_score_sel.values[0]))
print("Selected Feature index: " + str(sel_comb))

# Print the selected features
feature_names_sel = seg_data.columns[np.array(sel_comb)]
print(feature_names_sel)

# Evaluate on Test Data
X_train_CFS_FS = X_train[:,sel_comb]
X_test_CFS_FS = X_test[:,sel_comb]

print(X_train_CFS_FS.shape, y_train.shape)

kNN_CFS_FS = kNN.fit(X_train_CFS_FS.reshape(-1, 1), y_train)
y_pred = kNN_CFS_FS.predict(X_test_CFS_FS.reshape(-1, 1))
#acc_CFS_FS = accuracy_score(y_pred, y_test)
print(y_pred.shape, y_test.shape)
acc_CFS_FS = r2_score(y_pred, y_test)
cv_acc_CFS_FS = cross_val_score(kNN_CFS_FS, X_train_CFS_FS.reshape(-1, 1), y_train, cv=5)

print("X_Val on training selected features: {0:.3f}".format(cv_acc_CFS_FS.mean())) 
print("Hold Out testing selected features: {0:.3f}".format(acc_CFS_FS)) 

# Best First Search - CFS
# The stopping criteria for this implementation is where 5 consecutive 
# non-improving feature subsets are found.
Sel_feat = cfs(X_train, y_train)
Sel_feat = Sel_feat[Sel_feat!=-1]
print(Sel_feat)

# Print the names of the features selected
feature_names_sel = seg_data.columns[Sel_feat]
print(feature_names_sel)

# Find the merit score for the search space of the selected feature subsets
merit = []
cv_acc_CFS = []
for i in range(1,len(Sel_feat)+1):
    X_train_CFS = X_train[:,Sel_feat[0:i]]
    merit.insert(i, merit_calculation(X_train_CFS, y_train))
    kNN_CFS = kNN.fit(X_train_CFS, y_train)
    cv_acc_CFS.insert(i,cross_val_score(kNN_CFS, X_train_CFS, y_train, cv=5).mean())

print(merit)

# Plot merit score as features are added
f1 = plt.figure(dpi = 300)
plt.plot(feature_names_sel, merit)
plt.title("Correlation based Feature Selection (Segmentation)")
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Merit Score")
plt.tight_layout()
plt.show()

# Evaluate on test data
X_test_CFS = X_test[:,Sel_feat]

kNN_CFS = kNN.fit(X_train_CFS, y_train)
y_pred = kNN_CFS.predict(X_test_CFS)
#acc_CFS = accuracy_score(y_pred, y_test)
acc_CFS = r2_score(y_pred, y_test)
cv_acc_CFS = cross_val_score(kNN_CFS, X_train_CFS, y_train, cv=5)

print("X_Val on training selected features: {0:.3f}".format(cv_acc_CFS.mean())) 
print("Hold Out testing selected features: {0:.3f}".format(acc_CFS)) 

# Plot Results
fig, ax = plt.subplots(dpi = 300)
width = 0.2

options = ['All', 'CFS (+ additional features)', 'CFS (Highest Merit)']
#n_feat = [X_train.shape[1], X_train_CFS.shape[1], X_train_CFS_FS.shape[1]]
#print(X_train.shape[1],X_train_CFS.shape[1],X_train_CFS_FS.shape[1])
accs = [acc, acc_CFS, acc_CFS_FS]
xv = [cv_acc.mean(), cv_acc_CFS.mean(), cv_acc_CFS_FS.mean()]

y_pos = np.arange(len(options))

p1 = ax.bar(y_pos-width/2, xv, width, align='center', label = 'Train (X-val)',
            color=['blue','blue','blue'],alpha=0.5)

p2 = ax.bar(y_pos+width/2, accs , width, align='center', label = 'Test (Hold-out)',
            color=['g','g','g'],alpha=0.5)

ax.set_ylim([0.7, 1])
ax2 = ax.twinx()

#p3 = ax2.plot([0,1,2], n_feat, color = 'red', label = 'Feature Count', marker = 'x', ms = 10, linewidth=0)
#ax2.set_ylim([0, 20])

ax.grid(axis = 'y')

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc = 'upper right')

ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xticks(y_pos, options)
ax.set_ylabel('Accuracy')
ax2.set_ylabel('Feature Count')
plt.title("Segmentation Dataset")
plt.show()

## Example 2: CFS on Penguins dataset
#Peng_data = pd.read_csv('penguins.csv').drop(columns = ['Unnamed: 0'])
#print(Peng_data.shape)
#print(Peng_data.head())
#
#y = Peng_data.pop('species').values
#X_raw = Peng_data.values
#X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, random_state=2, test_size=1/2)
#
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_tr_raw)
#X_test = scaler.transform(X_ts_raw)
#max_length = X_train.shape[0]
#feat_num = X_train.shape[1]
#print(X_train.shape, X_test.shape)
#
##kNN = KNeighborsClassifier(n_neighbors=5)
#kNN = KNeighborsRegressor(n_neighbors=5)
#kNN = kNN.fit(X_train, y_train)
#y_pred = kNN.predict(X_test)
##acc = accuracy_score(y_pred, y_test)
#acc = r2_score(y_pred, y_test)
#cv_acc = cross_val_score(kNN, X_train, y_train, cv=8)
#
#print("X_Val on training all features: {0:.3f}".format(cv_acc.mean())) 
#print("Hold Out testing all features: {0:.3f}".format(acc)) 
#
## Forward Search - CFS
#merit_score_sel, sel_comb = CFS_FS(X_train, y_train)
#
#print("Selected Features Merit Score: " + str(merit_score_sel.values[0]))
#print("Selected Feature index: " + str(sel_comb))
#
## Evalutate on test data
#X_train_CFS_FS = X_train[:,sel_comb]
#X_test_CFS_FS = X_test[:,sel_comb]
#
#kNN_CFS_FS = kNN.fit(X_train_CFS_FS.reshape(-1, 1), y_train)
#y_pred = kNN_CFS_FS.predict(X_test_CFS_FS)
##acc_CFS_FS = accuracy_score(y_pred, y_test)
#acc_CFS_FS = r2_score(y_pred, y_test)
#cv_acc_CFS_FS = cross_val_score(kNN_CFS_FS, X_train_CFS_FS, y_train, cv=8)
#
#print("X_Val on training selected features: {0:.3f}".format(cv_acc_CFS_FS.mean())) 
#print("Hold Out testing selected features: {0:.3f}".format(acc_CFS_FS)) 
#
## Best First Search - CFS
#Sel_feat = cfs(X_train, y_train)
#Sel_feat = Sel_feat[Sel_feat!=-1]
#print(Sel_feat)
#
## Print the names of the features selected
#feature_names_sel = seg_data.columns[Sel_feat]
#print(feature_names_sel)
#
## Find the merit score for the search space of the selected feature subsets
#merit = []
#cv_acc_CFS = []
#for i in range(1,len(Sel_feat)+1):
#    X_train_CFS = X_train[:,Sel_feat[0:i]]
#    merit.insert(i, merit_calculation(X_train_CFS, y_train))
#    kNN_CFS = kNN.fit(X_train_CFS.reshape(-1, 1), y_train)
#    cv_acc_CFS.insert(i,cross_val_score(kNN_CFS, X_train_CFS, y_train, cv=8).mean())
#
#print(merit)
#
## Plot merit score as features are added
#f1 = plt.figure(dpi = 300)
#plt.plot(feature_names_sel, merit)
#plt.title("Correlation based Feature Selection (Penguins)")
#plt.xticks(rotation=90)
#plt.xlabel("Features")
#plt.ylabel("Merit Score")
#plt.tight_layout()
#
## Evaluate on Test data
#X_test_CFS = X_test[:, Sel_feat]
#
#kNN_CFS = kNN.fit(X_train_CFS.reshape(-1, 1), y_train)
#y_pred = kNN_CFS.predict(X_test_CFS.reshape(-1, 1))
##acc_CFS = accuracy_score(y_pred, y_test)
#acc_CFS = r2_score(y_pred, y_test)
#cv_acc_CFS = cross_val_score(kNN_CFS, X_train_CFS, y_train, cv=8)
#
#print("X_Val on training selected features: {0:.3f}".format(cv_acc_CFS.mean())) 
#print("Hold Out testing selected features: {0:.3f}".format(acc_CFS)) 
#
## Plot Results
#fig, ax = plt.subplots(dpi = 300)
#width = 0.2
#
#options = ['All', 'CFS (+ additional features)', 'CFS (Highest Merit)']
#n_feat = [X_train.shape[1], X_train_CFS.shape[1], X_train_CFS_FS.shape[1]]
#accs = [acc, acc_CFS, acc_CFS_FS]
#xv = [cv_acc.mean(), cv_acc_CFS.mean(), cv_acc_CFS_FS.mean()]
#
#y_pos = np.arange(len(options))
#
#p1 = ax.bar(y_pos-width/2, xv, width, align='center', label = 'Train (X-val)',
#            color=['blue','blue','blue'],alpha=0.5)
#
#p2 = ax.bar(y_pos+width/2, accs , width, align='center', label = 'Test (Hold-out)',
#            color=['g','g','g'],alpha=0.5)
#
#ax.set_ylim([0.7, 1])
#ax2 = ax.twinx()
#
#p3 = ax2.plot([0,1,2],n_feat, color = 'red', label = 'Feature Count', marker = 'x', ms = 10, linewidth=0)
#ax2.set_ylim([0, 5])
#
#ax.grid(axis = 'y')
#
#h1, l1 = ax.get_legend_handles_labels()
#h2, l2 = ax2.get_legend_handles_labels()
#ax2.legend(h1+h2, l1+l2, loc = 'lower right')
#
#ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
#
#plt.xticks(y_pos, options)
#ax.set_ylabel('Accuracy')
#ax2.set_ylabel('Feature Count')
#plt.title("Penguins Dataset")
#plt.show()
#
## References:
## 
## Jundong Li, Kewei Cheng, Suhang Wang, Fred Morstatter, Robert P. Trevino, 
## Jiliang Tang, and Huan Liu. 2017. Feature Selection: A Data Perspective. 
## ACM Comput. Surv. 50, 6, Article 94 (January 2018), 45 pages. 
## DOI:https://doi.org/10.1145/3136625
