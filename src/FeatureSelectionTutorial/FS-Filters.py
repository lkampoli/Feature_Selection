#!/usr/bin/env python
# coding: utf-8

# # Feature Selection using Filters
# ### Feature Scoring - two methods  
# 1. Chi square statistic
# 2. Information Gain

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
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
                                                       random_state=1, test_size=1/2)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)

feature_names = seg_data.columns
X_train.shape, X_test.shape


# ### Feature Scores  
# Determine the chi-squared and information gain scores for all features using the training set.   
# **Note:** The mutual information score returned by `mutual_info_classif` is effectively an information gain score.  

# In[5]:


chi2_score, pval = chi2(X_train, y_train)
chi2_score = np.nan_to_num(chi2_score)
chi2_score
# The chi square scores for the features


# In[6]:


i_scores = mutual_info_classif(X_train,y_train)
i_scores
# The i-gain scores for the features


# Store the scores in a dataframe indexed by the feature names.

# In[7]:


df=pd.DataFrame({'Mutual Info.':i_scores,'Chi Square':chi2_score,'Feature':feature_names})
df.set_index('Feature', inplace = True)
df.sort_values('Mutual Info.', inplace = True, ascending = False)
df


# ### Plotting the Filter scores
# We see that the two scores are fairly well correlated.  
# The Spearman correlation is 0.89.

# In[8]:


fig, ax = plt.subplots()
rr = range(0,len(feature_names))
ax2 = ax.twinx()
ax.plot(df.index, df["Mutual Info."], label='I-Gain')
ax2.plot(df.index, df["Chi Square"], color='skyblue', label='Chi Squared')
ax.set_xticks(rr)

ax.set_xticklabels(list(df.index), rotation = 90)
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('I-Gain')
ax2.set_ylabel('Chi Squared')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)


# In[9]:


from scipy import stats
stats.spearmanr(chi2_score, i_scores)


# ## Feature Selection
# Compare  
# - Baseline: all features
# - Top three, I-Gain and Chi-Square
# - Top six, I-Gain and Chi-Square
# - Top half (12), I-Gain and Chi-Square

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# ### Baseline Classifier

# In[11]:


model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred,y_test)
acc


# In[12]:


n_features = X_train.shape[1]
n_features


# In[13]:


filters = [mutual_info_classif, chi2]
k_options = [n_features, 3, 6, 10, 15]
filt_scores = {}
chi_scores = {}
i_gain_scores = {}

for the_filter in filters:
    accs = []
    for k_val in k_options:
        FS_trans = SelectKBest(the_filter, 
                           k=k_val).fit(X_train, y_train)
        X_tR_new = FS_trans.transform(X_train)
        X_tS_new = FS_trans.transform(X_test)

        model.fit(X_tR_new, y_train)

        y_tS_pred = model.predict(X_tS_new)
        
        acc = accuracy_score(y_test, y_tS_pred)
        accs.append(acc)
        print(the_filter, k_val, acc)
    filt_scores[the_filter.__name__] = accs


# In[14]:


import matplotlib.pyplot as plt 
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots()
width = 0.3
sb = 'skyblue'

options = ['All'] + k_options[1:]
ig = filt_scores['mutual_info_classif']
ch = filt_scores['chi2']

y_pos = np.arange(len(options))

p1 = ax.bar(y_pos-width, ig, width, align='center', 
            color=['red', 'blue', 'blue','blue','blue'],alpha=0.5)
p2 = ax.bar(y_pos, ch, width, align='center', 
            color=['red', sb, sb, sb, sb],alpha=0.5)

ax.legend((p1[1], p2[1]), ('I-Gain', 'Chi Squared'),loc='lower right')
ax.set_ylim([0.5, 1])
plt.grid(axis = 'y')
plt.yticks(np.arange(0.5,1.05,0.1))

plt.xticks(y_pos, options)
plt.ylabel('Test Set Accuracy')
plt.xlabel('Feature Counts')
plt.show()


# ## Hybrid Filter Wrapper Strategy
# We rank the features using information gain (well mutual information) and select the _k_ best to build a classifier.  
# We iterate through increasing values of *k*.  
# `SelectKBest` is a _transform_ that transforms the training data.
# 

# In[ ]:


cv_acc_scores = []
tst_acc_scores = []
best_acc = 0
best_k = 0
for kk in range(1, X_train.shape[1]+1):
    FS_trans = SelectKBest(mutual_info_classif, 
                           k=kk).fit(X_train, y_train)
    X_tR_new = FS_trans.transform(X_train)
    X_tS_new = FS_trans.transform(X_test)
    cv_acc = cross_val_score(model, X_tR_new, y_train, cv=8)
    cv_acc_scores.append(cv_acc.mean())
    y_pred_temp = model.fit(X_tR_new, y_train).predict(X_tS_new)
    tst_acc_scores.append(accuracy_score(y_pred_temp, y_test))
    if cv_acc.mean() > best_acc:
        best_acc = cv_acc.mean()
        best_k = kk
df['Training Acc.'] = cv_acc_scores
df['Test Acc.'] = tst_acc_scores

print(best_k, best_acc)
df.head(15)


# In[ ]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

n = len(df.index)
rr = range(0,n)
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(df.index, df["Mutual Info."], label='I-Gain',width=.35)

ax2.plot(df.index, df["Training Acc."], color='green', label='Training Acc.')
ax2.plot(df.index, df["Test Acc."], color='lightgreen', label='Test Acc')
ax.set_xticks(rr)
ax2.plot(best_k-1,best_acc,'gx') 
ax.set_xticklabels(list(df.index), rotation = 90)
ax.set_xlabel('Features')
ax.set_ylabel('I-Gain')
ax2.set_ylabel('Accuracy')
fig.legend(loc="upper right", bbox_to_anchor=(1,0.8), bbox_transform=ax.transAxes)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




