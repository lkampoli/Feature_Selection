#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis
# PCA using the PCA implementaiton in `scikit-learn`

# In[1]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ## Top Trumps
# `HarryPotterTT.csv` contains data on Top Trumps cards.  
# There are 22 examples described by 5 features.

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
TT_df = pd.read_csv('HarryPotterTT.csv')
TT_df


# Extract the data into a numpy array X.  
# And scale the data.

# In[3]:


y = TT_df.pop('Name').values
X = TT_df.values
X_scal = StandardScaler().fit_transform(X)
X.shape


# Apply PCA.

# In[4]:


pcaHP = PCA(n_components=4)
X_r = pcaHP.fit(X_scal).transform(X_scal)
pcaHP.explained_variance_ratio_


# There are five features being projected onto 4 PCs so the projection matrix is 4 x 5.

# In[5]:


pcaHP.components_


# In[6]:


df = pd.DataFrame(pcaHP.explained_variance_ratio_, 
                  index=['PC1','PC2','PC3','PC4'],columns =['var'])

pl = df.plot.bar(color='red',figsize=(5,4))
pl.set_ylabel("Variance Explained")
pl.set_ylim([0,0.8])


# In[7]:


plt.figure(figsize=(8,6))
lw = 2
labels = list(range(len (y)))
labels[0]='Harry'
labels[1]='Hermione'
labels[3]='Prof D'
labels[5]='Prof McG'
labels[6]='Prof Moody'
labels[18]='Cedric D'
labels[19]='Viktor K'
labels[21]='Lucius Malfoy'
labels[4]='Snape'
labels[12]='Draco Malfoy'

plt.scatter(X_r[:, 0], X_r[:, 1])

for label, xi, yi in zip(labels, X_r[:, 0], X_r[:, 1]):
    plt.annotate(
        label,
        xy=(xi, yi), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')

plt.xlabel('PC1 (49%)')
plt.ylabel('PC2 (32%)')
plt.title('PCA of HP dataset')

plt.show()


# ## Comment
#  - This plot shows the data projected onto the first 2 PCs.  
#  - These PCs account for 81% of the variance in the data. 
#  - It might be argued that the first PC captures *competence* and the second represents *malevolence*. 
# 
