#!/usr/bin/env python
# coding: utf-8

# Random Forest Feature Importance
# As a side-effect of buiding a random forest ensemble, we get a very useful estimate of feature importance. 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt 

# Segmentation Data
#seg_data = pd.read_csv('segmentation-all.csv')
#seg_data = pd.read_csv('../../data/STS/transport/boltzmann/shear_viscosity.csv')
seg_data = pd.read_csv('../../data/MT/DB6T.csv')
print(seg_data.shape)
print(seg_data.head())

#print(seg_data['Class'].value_counts())
print(seg_data['Viscosity'].value_counts())

# Load the data, scale it and divide into train and test sets.  
# The filters are *trained* using the training data and then a classifier is trained on the feature subset and tested on the test set. 
#y = seg_data.pop('Class').values
y = seg_data.pop('Viscosity').values

X_raw = seg_data.values

#X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, random_state=1, test_size=1/2)
X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, train_size=0.75, test_size=0.25, random_state=666)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)

feature_names = seg_data.columns
print(X_train.shape, X_test.shape)

# Build the Random Forest and calculate the scores.  
n_trees = 1000
#RF = RandomForestClassifier(n_estimators=n_trees, max_depth=2, random_state=0)
RF = RandomForestRegressor(n_estimators=n_trees, max_depth=2, random_state=0)
print(RF.fit(X_train,y_train))

rf_scores = RF.feature_importances_
print(rf_scores)

# Calculate the I-gain scores for comparison.
#i_scores = mutual_info_classif(X_train, y_train)
i_scores = mutual_info_regression(X_train, y_train)
print(i_scores) # The i-gain scores for the features

df=pd.DataFrame({'Mutual Info.':i_scores,'RF Score':rf_scores,'Feature':feature_names})
df.set_index('Feature', inplace = True)
df.sort_values('Mutual Info.', inplace = True, ascending = False)
print(df)

# Plotting the two sets of scores
n = len(df.index)
rr = range(0,n)
fig, ax = plt.subplots(figsize=(6,5))
ax2 = ax.twinx()
ax.bar(df.index, df["RF Score"], label='RF Score',width=.35, color = 'g')

ax2.set_xticks(rr)
ax2.plot(df.index, df["Mutual Info."], label='I-Gain', color = 'navy')

ax.set_xticklabels(list(df.index), rotation = 90)
ax.set_xlabel('Features')
ax.set_ylabel('I-Gain')
ax2.set_ylabel('RF Score')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.show()

from scipy import stats
print(stats.spearmanr(rf_scores, i_scores))

## Penguins
#penguins_df = pd.read_csv('penguins.csv', index_col = 0)
#
#feature_names = penguins_df.columns
#print(penguins_df.shape)
#print(penguins_df.head())
#
#y = penguins_df.pop('species').values
#X = penguins_df.values
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                       random_state=1, test_size=1/2)
#feature_names = penguins_df.columns
#X_train.shape, X_test.shape
#
#RF = RandomForestClassifier(n_estimators=n_trees, max_depth=2, random_state=0)
#RF.fit(X_train,y_train)
#
#rf_scores = RF.feature_importances_
#print(rf_scores)
#print(feature_names)
#
#i_scores = mutual_info_classif(X_train,y_train)
#print(i_scores) # The i-gain scores for the features
#
#pen_df=pd.DataFrame({'Mutual Info.':i_scores,'RF Score':rf_scores,'Feature':feature_names})
#pen_df.set_index('Feature', inplace = True)
#pen_df.sort_values('Mutual Info.', inplace = True, ascending = False)
#print(pen_df)
#
#n = len(pen_df.index)
#rr = range(0,n)
#fig, ax = plt.subplots(figsize=(2.5,5))
#ax2 = ax.twinx()
#ax.bar(pen_df.index, pen_df["RF Score"], label='RF Score',width=.35, color = 'g')
#
#ax2.set_xticks(rr)
#ax2.plot(pen_df.index, pen_df["Mutual Info."], label='I-Gain', color = 'navy')
#
#ax.set_xticklabels(list(pen_df.index), rotation = 90)
#ax.set_xlabel('Features')
#ax.set_ylabel('I-Gain')
#ax2.set_ylabel('RF Score')
#fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
#plt.show()
#
#stats.spearmanr(rf_scores, i_scores)
