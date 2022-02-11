#!/usr/bin/env python
# coding: utf-8

# # Feature Selection using D-Trees
# Feature selection is implicit in the construction of decision trees as, typically not all features will appear in the tree.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt 


# In[2]:


penguins_df = pd.read_csv('penguins.csv', index_col = 0)

feature_names = penguins_df.columns
print(penguins_df.shape)
penguins_df.head()


# In[3]:


penguins_df['species'].value_counts()


# Load the data into numpy arrays and divide into train and test sets.  
# The filters are *trained* using the training data and then a classifier is trained on the feature subset and tested on the test set.  
# With D-Trees there is no need to scale the data.

# In[4]:


y = penguins_df.pop('species').values
X = penguins_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                       random_state=1, test_size=1/2)
feature_names = penguins_df.columns
X_train.shape, X_test.shape


# ## Full Tree  
# Tree with no pruning

# In[5]:


ftree = DecisionTreeClassifier(criterion='entropy')
ftree = ftree.fit(X_train, y_train)
y_pred = ftree.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print("Test set accuract %4.2f" % (acc))


# In[6]:


plt.figure(figsize=(11,6))

tree.plot_tree(ftree, fontsize = 10, feature_names = feature_names,
                      class_names=['Adelie','Gentoo', 'Chinstrap'], 
                      label = 'none', filled=True, impurity = False,
               rounded=True) 


# In[7]:


fi = ftree.feature_importances_

for fi_val, f_name in zip(fi,feature_names):
    print(" %4.2f  %s" % (fi_val, f_name))


# How many leaves in this tree?

# In[8]:


ftree.get_n_leaves()


# ### Pruned Tree
# Ok, build a tree with just 3 leaves.

# In[9]:


p_tree = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes = 3)
p_tree = p_tree.fit(X_train, y_train)
y_pred = p_tree.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print("Test set accuract %4.2f" % (acc))


# In[10]:


plt.figure(figsize=(11,6))

tree.plot_tree(p_tree, fontsize = 10, feature_names = feature_names,
                      class_names=['Adelie','Gentoo', 'Chinstrap'], 
                      label = 'none', filled=True, impurity = False,
               rounded=True) 


# Now two of the features are not selected. 

# In[11]:


fi = p_tree.feature_importances_
for fi_val, f_name in zip(fi,feature_names):
    print(" %4.2f  %s" % (fi_val, f_name))

