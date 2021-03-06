#!/usr/bin/env python
# coding: utf-8

# Feature Selection Using LASSO
# The `scikit-learn` Logistic Regression includes regularization.  
# If the penalty/loss is set to L1 this is effectively LASSO.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from matplotlib.ticker import MaxNLocator
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import linear_model
import matplotlib.pyplot as plt

# Segmentation dataset
#seg_data = pd.read_csv('segmentation-all.csv')
#seg_data = pd.read_csv('../../data/STS/transport/boltzmann/shear_viscosity.csv')
seg_data = pd.read_csv('../../data/MT/DB6T.csv')
print(seg_data.shape)
print(seg_data.head())

#print(seg_data['Class'].unique())
print(seg_data['Viscosity'].value_counts())

# Reduce the data to just two classes. This makes the feature selection process easier to follow. 
#seg_data2C = seg_data.loc[seg_data['Class'].isin(['WINDOW','CEMENT'])]

#y = seg_data2C.pop('Class').values
y = seg_data.pop('Viscosity').values
#X_raw = seg_data2C.values
X_raw = seg_data.values

#feature_names = seg_data2C.columns
feature_names = seg_data.columns

#X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, random_state=42, test_size=1/2)
X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, train_size=0.075, test_size=0.025, random_state=666, shuffle=True)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)
max_k = X_train.shape[1]
print(X_train.shape, X_test.shape)

classes=np.unique(y_train)
print(classes)

# Logistic Regression
#lr = LogisticRegression(solver='saga', penalty='none', max_iter=5000)
lr = linear_model.Lasso(alpha=0.0, max_iter=100)
lr_tr = lr.fit(X_train, y_train)
full_acc = lr_tr.score(X_test, y_test)
full_tr_acc = cross_val_score(lr, X_train, y_train, cv=5)
print('Test Accuracy: {:.2f}'.format(full_acc), 'Training Accuracy: {:.2f}'.format(full_tr_acc.mean()))

betas = np.absolute(lr_tr.coef_[0])
f_scores=pd.DataFrame({'No Lasso':betas,'Feature':feature_names})
f_scores.set_index('Feature', inplace = True)
print(f_scores)

# Lasso feature selection
# Logistic regression with L1 regularisation.   
# `SelectFromModel` will select the top features out of `max_features`   
# The `C` parameter in `LogisticRegression` is the regularisation parameter, smaller values means stronger regularisation, default is 1.       
# You can select a specific number of features from `SelectFromModel` using the `max_features` parameter
# 
# Using default regularisation
#lasso = SelectFromModel(LogisticRegression(penalty="l1", C=1, solver="saga", max_iter=1000), max_features=X_train.shape[1])
lasso = SelectFromModel(linear_model.Lasso(alpha=0.000000003, max_iter=100), max_features=X_train.shape[1])
lasso.fit(X_train, y_train)
#lasso_def_features = list(seg_data2C.columns[lasso.get_support()])
lasso_def_features = list(seg_data.columns[lasso.get_support()])
print('Selected features:', lasso_def_features)

f_scores['Lasso C=1'] = np.absolute(lasso.estimator_.coef_[0])

# Reduce the data to just the selected features
X_tr_def = lasso.transform(X_train)
X_tst_def = lasso.transform(X_test)
X_tr_def.shape, X_tst_def.shape

#lr = LogisticRegression(solver='saga', penalty='none', max_iter=3000)
lr = linear_model.Lasso(alpha=0.0, max_iter=100)
lr_def = lr.fit(X_tr_def, y_train)
default_acc = lr_def.score(X_tst_def, y_test)
def_tr_acc = cross_val_score(lr, X_tr_def, y_train, cv=5)
print('Lasso C=1 selects %d features' % (X_tr_def.shape[1]))
print('Lasso C=1  Test Accuracy: {:.2f}'.format(default_acc), 'Lasso C=1  Train Accuracy (x-val): {:.2f}'.format(def_tr_acc.mean()))

# Using less regularisation
#lasso_mild = SelectFromModel(LogisticRegression(penalty="l1", C=10, solver="saga", max_iter=3000), max_features=X_train.shape[1])
lasso_mild = SelectFromModel(linear_model.Lasso(alpha=0.0, max_iter=100), max_features=X_train.shape[1])
lasso_mild.fit(X_train, y_train)
#lasso_mild_features = list(seg_data2C.columns[lasso_mild.get_support()])
lasso_mild_features = list(seg_data.columns[lasso_mild.get_support()])
print('Selected features:', lasso_mild_features)

f_scores['Lasso C=10'] = np.absolute(lasso_mild.estimator_.coef_[0])

X_tr_mild = lasso_mild.transform(X_train)
X_tst_mild = lasso_mild.transform(X_test)
X_tr_mild.shape, X_tst_mild.shape

#lr = LogisticRegression(solver='saga', penalty='none', max_iter=4000)
lr = LinearRegression()
lr_mild = lr.fit(X_tr_mild, y_train)
mild_acc = lr_mild.score(X_tst_mild, y_test)
mild_tr_acc = cross_val_score(lr, X_tr_mild, y_train, cv=8)
print('Lasso C=10  selects %d features' % (X_tr_mild.shape[1]))
print('Lasso C=10 Test Accuracy: {:.2f}'.format(mild_acc), 
      'Lasso C=10 Train Accuracy (x-val): {:.2f}'.format(mild_tr_acc.mean()))

# Plotting results
print(f_scores.head())
df_plot = f_scores.sort_values('No Lasso', ascending = False)
ax = df_plot[['No Lasso','Lasso C=10','Lasso C=1']].plot.bar(figsize=(7,4))
ax.grid( axis = 'y')
plt.ylabel('Betas (Abs. Val.)')

fig, ax = plt.subplots()
width = 0.2

options = ['No Lasso', 'Lasso C=1', 'Lasso C+10']
n_feat = [X_train.shape[1], len(lasso_def_features), len(lasso_mild_features)]
accs = [full_acc,default_acc,mild_acc]
xv = [full_tr_acc.mean(), def_tr_acc.mean(), mild_tr_acc.mean()]

y_pos = np.arange(len(options))

p1 = ax.bar(y_pos-width/2, xv, width, align='center', label = 'Train (X-val)', color=['blue','blue','blue'],alpha=0.5)
p2 = ax.bar(y_pos+width/2, accs , width, align='center', label = 'Test (Hold-out)', color=['g','g','g'],alpha=0.5)

ax.set_ylim([0.7, 1])
ax2 = ax.twinx()

p3 = ax2.plot([0,1,2],n_feat, color = 'red', label = 'Feature Count', marker = 'x', ms = 10, linewidth=0)
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

## Penguins dataset
## Lasso feature selection on the Penguins dataset
#penguins = pd.read_csv('penguins.csv', index_col = 0)
#print(penguins.shape)
#print(penguins.head())
#
#classes=np.unique(penguins['species'])
#print(classes)
#
## Reduce to a 2-class dataset to make Lasso feature selection more transparent. 
#penguins2C = penguins.loc[penguins['species'].isin(['Adelie','Chinstrap'])]
#
#y = penguins2C.pop('species').values
#X_raw = penguins2C.values
#feature_names = penguins2C.columns
#X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, random_state=2, test_size=1/2)
#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_tr_raw)
#X_test = scaler.transform(X_ts_raw)
#max_k = X_train.shape[1]
#print(X_train.shape, X_test.shape)
#
## Logistic Regression
#lr = LogisticRegression(solver='saga', penalty='none', max_iter=5000)
#lr_tr = lr.fit(X_train, y_train)
#full_acc = lr_tr.score(X_test, y_test)
#full_tr_acc = cross_val_score(lr, X_train, y_train, cv=8)
#print('Test Accuracy: {:.2f}'.format(full_acc), 
#      'Training Accuracy: {:.2f}'.format(full_tr_acc.mean()))
#
#betas = np.absolute(lr_tr.coef_[0])
#f_scores=pd.DataFrame({'No Lasso':betas,'Feature':feature_names})
#f_scores.set_index('Feature', inplace = True)
#print(f_scores)
#
## Lasso feature selection
## Logistic regression with L1 regularisation.   
## `SelectFromModel` will select the top features out of `max_features`   
## The `C` parameter in `LogisticRegression` is the regularisation parameter, smaller values means stronger regularisation, default is 1.       
## You can select a specific number of features from `SelectFromModel` using the `max_features` parameter
## 
## Using default regularisation
#lasso = SelectFromModel(LogisticRegression(penalty="l1", 
#                     C=1, solver="saga", max_iter=1000), max_features=X_train.shape[1])
#lasso.fit(X_train, y_train)
#lasso_def_features = list(penguins2C.columns[lasso.get_support()])
#print('Selected features:', lasso_def_features)
#
#f_scores['Lasso C=1'] = np.absolute(lasso.estimator_.coef_[0])
#
## Reduce the data to just the selected features
#X_tr_def = lasso.transform(X_train)
#X_tst_def = lasso.transform(X_test)
#X_tr_def.shape, X_tst_def.shape
#
#lr = LogisticRegression(solver='saga', penalty='none', max_iter=4000)
#lr_def = lr.fit(X_tr_def, y_train)
#default_acc = lr_def.score(X_tst_def, y_test)
#def_tr_acc = cross_val_score(lr, X_tr_def, y_train, cv=8)
#print('Lasso C=1 selects %d features' % (X_tr_def.shape[1]))
#print('Lasso C=1 Test Accuracy: {:.2f}'.format(default_acc), 
#      'Lasso C=1 Train Accuracy (x-val): {:.2f}'.format(def_tr_acc.mean()))
#
## Using less regularisation
#lasso_mild = SelectFromModel(LogisticRegression(penalty="l1", 
#                     C=10, solver="saga", max_iter=3000), max_features=X_train.shape[1])
#lasso_mild.fit(X_train, y_train)
#lasso_mild_features = list(penguins2C.columns[lasso_mild.get_support()])
#print('Selected features:', lasso_mild_features)
#
#f_scores['Lasso C=10'] = np.absolute(lasso_mild.estimator_.coef_[0])
#
#X_tr_mild = lasso_mild.transform(X_train)
#X_tst_mild = lasso_mild.transform(X_test)
#X_tr_mild.shape, X_tst_mild.shape
#
#lr = LogisticRegression(solver='saga', penalty='none', max_iter=4000)
#lr_mild = lr.fit(X_tr_mild, y_train)
#mild_acc = lr_mild.score(X_tst_mild, y_test)
#mild_tr_acc = cross_val_score(lr, X_tr_mild, y_train, cv=8)
#print('Lasso C=10 selects %d features' % (X_tr_mild.shape[1]))
#print('Lasso C=10 Test Accuracy: {:.2f}'.format(mild_acc), 
#      'Lasso C=10 Train Accuracy (x-val): {:.2f}'.format(mild_tr_acc.mean()))
#
## Plotting results
#f_scores.head()
#
#df_plot = f_scores.sort_values('No Lasso', ascending = False)
#ax = df_plot[['No Lasso','Lasso C=10','Lasso C=1']].plot.bar(figsize=(3,4))
#ax.grid( axis = 'y')
#plt.ylabel('Betas (Abs. Val.)')
#
#fig, ax = plt.subplots()
#width = 0.2
#
#options = ['No Lasso', 'Lasso C=1', 'Lasso C=10']
#n_feat = [X_train.shape[1], len(lasso_def_features), len(lasso_mild_features)]
#accs = [full_acc,default_acc,mild_acc]
#xv = [full_tr_acc.mean(), def_tr_acc.mean(), mild_tr_acc.mean()]
#
#y_pos = np.arange(len(options))
#
#p1 = ax.bar(y_pos-width/2, xv, width, align='center', label = 'Train (X-val)',
#            color=['blue','blue','blue'],alpha=0.5)
#p2 = ax.bar(y_pos+width/2, accs , width, align='center', label = 'Test (Hold-out)',
#            color=['g','g','g'],alpha=0.5)
#
#ax.set_ylim([0.7, 1])
#ax2 = ax.twinx()
#
#p3 = ax2.plot([0,1,2],n_feat, color = 'red', label = 'Feature Count',
#              marker = 'x', ms = 10, linewidth=0)
#ax2.set_ylim([0, 5])
#
#ax.grid(axis = 'y')
#
#h1, l1 = ax.get_legend_handles_labels()
#h2, l2 = ax2.get_legend_handles_labels()
#ax2.legend(h1+h2, l1+l2, loc='lower right')
#
#ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
#
#plt.xticks(y_pos, options)
#ax.set_ylabel('Accuracy')
#ax2.set_ylabel('Feature Count')
#
#plt.show()
