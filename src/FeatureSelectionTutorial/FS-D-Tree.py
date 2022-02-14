#!/usr/bin/env python
# coding: utf-8

# Feature Selection using D-Trees
# Feature selection is implicit in the construction of decision trees as, typically not all features will appear in the tree.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt 

#penguins_df = pd.read_csv('penguins.csv', index_col = 0)
penguins_df = pd.read_csv('../../data/STS/transport/boltzmann/shear_viscosity.csv')

feature_names = penguins_df.columns
print(feature_names)
print(penguins_df.shape)
print(penguins_df.head())

#print(penguins_df['species'].value_counts())
print(penguins_df['Viscosity'].value_counts())

# Load the data into numpy arrays and divide into train and test sets.  
# The filters are *trained* using the training data and then a classifier is trained on the feature subset and tested on the test set.  
# With D-Trees there is no need to scale the data.

#y = penguins_df.pop('species').values
y = penguins_df.pop('Viscosity').values
X = penguins_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=1/2)
feature_names = penguins_df.columns
print(X_train.shape, X_test.shape)

# Full Tree  
# Tree with no pruning
#ftree = DecisionTreeClassifier(criterion='entropy')
ftree = DecisionTreeRegressor() # default settings
ftree = ftree.fit(X_train, y_train)
y_pred = ftree.predict(X_test)
#acc = accuracy_score(y_pred,y_test)
acc = r2_score(y_test,y_pred)
#acc = mean_squared_error(y_pred,y_test)
#acc = mean_absolute_error(y_test,y_pred)
print("Test set accuracy %4.2f" % (acc))

plt.figure(figsize=(11,6))

tree.plot_tree(ftree, fontsize = 10, feature_names = feature_names,
#              class_names=['Adelie','Gentoo', 'Chinstrap'], 
               label = 'none', filled=True, impurity = False,
               rounded=True) 
plt.show()

fi = ftree.feature_importances_
for fi_val, f_name in zip(fi,feature_names):
    print(" %4.2f  %s" % (fi_val, f_name))

# How many leaves in this tree?
print(ftree.get_n_leaves())

# Pruned Tree
# Ok, build a tree with just 3 leaves.
#p_tree = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes = 3)
p_tree = DecisionTreeRegressor(max_leaf_nodes = 3)
p_tree = p_tree.fit(X_train, y_train)
y_pred = p_tree.predict(X_test)
#acc = accuracy_score(y_pred,y_test)
acc = r2_score(y_test,y_pred)
#acc = mean_absolute_error(y_pred,y_test)
#acc = mean_absolute_error(y_test,y_pred)
print("Test set accuracy %4.2f" % (acc))

plt.figure(figsize=(11,6))

tree.plot_tree(p_tree, fontsize = 10, feature_names = feature_names,
#              class_names=['Adelie','Gentoo', 'Chinstrap'], 
               label = 'none', filled=True, impurity = False,
               rounded=True) 
plt.show()

# Now two of the features are not selected. 
fi = p_tree.feature_importances_
for fi_val, f_name in zip(fi,feature_names):
    print(" %4.2f  %s" % (fi_val, f_name))
