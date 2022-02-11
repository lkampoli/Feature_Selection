#!/usr/bin/env python
# coding: utf-8

# # Random Forest Feature Importance
# As a side-effect of buiding a random forest ensemble, we get a very useful estimate of feature importance. 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
#get_ipython().run_line_magic('matplotlib', 'inline')


# ### Segmentation Data

# In[2]:


seg_data = pd.read_csv('segmentation-all.csv')
print(seg_data.shape)
seg_data.head()


# In[3]:


seg_data['Class'].value_counts()


# Load the data, scale it and divide into train and test sets.  
# The filters are *trained* using the training data and then a classifier is trained on the feature subset and tested on the test set. 

# In[4]:


y = seg_data.pop('Class').values
X_raw = seg_data.values

X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, 
                                                       random_state=1, test_size=1/2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)

feature_names = seg_data.columns
X_train.shape, X_test.shape


# Build the Random Forest and calculate the scores.  

# In[ ]:


n_trees = 1000
RF = RandomForestClassifier(n_estimators=n_trees, max_depth=2, random_state=0)
RF.fit(X_train,y_train)


# In[ ]:


rf_scores = RF.feature_importances_
rf_scores


# Calculate the I-gain scores for comparison.

# In[ ]:


i_scores = mutual_info_classif(X_train,y_train)
i_scores
# The i-gain scores for the features


# In[ ]:


df=pd.DataFrame({'Mutual Info.':i_scores,'RF Score':rf_scores,'Feature':feature_names})
df.set_index('Feature', inplace = True)
df.sort_values('Mutual Info.', inplace = True, ascending = False)
df


# Plotting the two sets of scores

# In[ ]:


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


# In[ ]:


from scipy import stats
stats.spearmanr(rf_scores, i_scores)


# ## Penguins

# In[ ]:


penguins_df = pd.read_csv('penguins.csv', index_col = 0)

feature_names = penguins_df.columns
print(penguins_df.shape)
penguins_df.head()


# In[ ]:


y = penguins_df.pop('species').values
X = penguins_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                       random_state=1, test_size=1/2)
feature_names = penguins_df.columns
X_train.shape, X_test.shape


# In[ ]:


RF = RandomForestClassifier(n_estimators=n_trees, max_depth=2, random_state=0)
RF.fit(X_train,y_train)


# In[ ]:


rf_scores = RF.feature_importances_
rf_scores


# In[ ]:


feature_names


# In[ ]:


i_scores = mutual_info_classif(X_train,y_train)
i_scores
# The i-gain scores for the features


# In[ ]:


pen_df=pd.DataFrame({'Mutual Info.':i_scores,'RF Score':rf_scores,'Feature':feature_names})
pen_df.set_index('Feature', inplace = True)
pen_df.sort_values('Mutual Info.', inplace = True, ascending = False)
pen_df


# In[ ]:


n = len(pen_df.index)
rr = range(0,n)
fig, ax = plt.subplots(figsize=(2.5,5))
ax2 = ax.twinx()
ax.bar(pen_df.index, pen_df["RF Score"], label='RF Score',width=.35, color = 'g')

ax2.set_xticks(rr)
ax2.plot(pen_df.index, pen_df["Mutual Info."], label='I-Gain', color = 'navy')

ax.set_xticklabels(list(pen_df.index), rotation = 90)
ax.set_xlabel('Features')
ax.set_ylabel('I-Gain')
ax2.set_ylabel('RF Score')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.show()


# In[ ]:


stats.spearmanr(rf_scores, i_scores)

