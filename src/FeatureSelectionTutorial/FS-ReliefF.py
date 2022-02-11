#!/usr/bin/env python
# coding: utf-8

# # Feature Selection using ReliefF
# https://epistasislab.github.io/scikit-rebate/using/
# `pip install skrebate`

# In[1]:


import pandas as pd
import numpy as np
from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


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
                                                       random_state=42, test_size=1/2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)

feature_names = seg_data.columns
X_train.shape, X_test.shape


# ### ReliefF
# - `ReliefF` will produce scores for all features.
# - `n_features_to_select` controls the transform behaviour, if a dataset is transformed this number of features will be retained. 

# In[5]:


reliefFS = ReliefF(n_features_to_select=11, n_neighbors=100, n_jobs = -1)


# In[6]:


reliefFS.fit(X_train,y_train)


# In[7]:


relief_scores = reliefFS.feature_importances_


# In[8]:


reliefFS.transform(X_train).shape


# Also calcuate I-Gain scores: to be used for comparision.

# In[9]:


i_scores = mutual_info_classif(X_train,y_train)
i_scores
# The i-gain scores for the features


# In[10]:


from scipy import stats
stats.spearmanr(relief_scores, i_scores)


# Store the ReliefF and I-Gain scores in a dataframe.  
# **Note:** The mutual information score returned by `mutual_info_classif` is effectively an information gain score.  

# In[11]:


df=pd.DataFrame({'Mutual Info.':i_scores,'ReliefF':relief_scores,'Feature':feature_names})
df.set_index('Feature', inplace = True)
df.sort_values('Mutual Info.', inplace = True, ascending = False)
df


# ### Plotting the ReliefF and I-Gain scores
# We see that the two scores are fairly well correlated.  
# The Spearman correlation is 0.81.

# In[12]:


fig, ax = plt.subplots()
rr = range(0,len(feature_names))
ax2 = ax.twinx()
ax.plot(df.index, df["Mutual Info."], label='I-Gain')
ax2.plot(df.index, df["ReliefF"], color='red', label='Relief')
ax.set_xticks(rr)

ax.set_xticklabels(list(df.index), rotation = 90)
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('I-Gain')
ax2.set_ylabel('ReliefF')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)


# ## Feature Selection
# Compare  
# - Baseline: all features
# - Visual inspection of the ReliefF plot suggests we select the top 11 features. 

# ### Baseline Classifier

# In[13]:


model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc_all = accuracy_score(y_pred,y_test)
acc_all


# In[14]:


n_features = X_train.shape[1]
n_features


# ### After feature selection
# We produce a reduced dataset with the 11 top ranking features selected by ReliefF

# In[15]:


X_tr_relief = reliefFS.transform(X_train)
X_ts_relief = reliefFS.transform(X_test)
X_tr_relief.shape


# In[16]:


kNN_relief = model.fit(X_tr_relief,y_train)
y_pred = kNN_relief.predict(X_ts_relief)
acc_11 = accuracy_score(y_pred,y_test)
acc_11


# In[17]:


import matplotlib.pyplot as plt 
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(2.5,3.5))
width = 0.5
sb = 'skyblue'

options = ['All', 'ReliefF 11']
scores = [acc_all,acc_11]

y_pos = np.arange(len(options))

p1 = ax.bar(y_pos, scores, width, align='center', 
            color=['red', 'blue'],alpha=0.5)

ax.set_ylim([0.5, 1])
plt.grid(axis = 'y')
plt.yticks(np.arange(0.5,1.05,0.05))
ax.text(0,acc_all, '%0.3f' % acc_all, ha='center', va = 'top')
ax.text(1,acc_11, '%0.3f' % acc_11, ha='center',va = 'top')

plt.xticks(y_pos, options)
plt.ylabel('Test Set Accuracy')
plt.xlabel('Features')
plt.show()


# In[ ]:




