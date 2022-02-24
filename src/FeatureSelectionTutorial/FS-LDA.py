#!/usr/bin/env python
# coding: utf-8

# Linear Discriminant Analysis + comparison with PCA
# Linear Discriminant Analysis using the LDA implementation in `scikit-learn`.
# The objective with LDA is to project the data into a reduced dimension space that maximises between-class separation.  
# PCA is also included for the purpose of comparison.  

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

# Penguins
penguins_df = pd.read_csv('penguins.csv', index_col = 0)
y = penguins_df.pop('species').values
X_raw = penguins_df.values

X_tr_raw, X_ts_raw, y_train, y_test = train_test_split(X_raw, y, random_state=1, test_size=1/2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_tr_raw)
X_test = scaler.transform(X_ts_raw)

feature_names = penguins_df.columns
print(penguins_df.shape)
print(penguins_df.head())

types = list(Counter(y).keys())
print(types)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
X_tr_lda = lda.transform(X_train)
print(X_tr_lda.shape)
print(lda.explained_variance_ratio_)

plt.figure(figsize=(4,4))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, target_name in zip(colors, types):
    plt.scatter(X_tr_lda[y_train == target_name, 0], X_tr_lda[y_train == target_name, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PC1 (84%)')
plt.ylabel('PC2 (16%)')
#plt.title('LDA of the Penguins dataset')
plt.show()

y_pred = lda.predict(X_test)
print(accuracy_score(y_pred,y_test))

# PCA
pca = PCA(n_components=4)
X_tr_pca = pca.fit(X_train).transform(X_train)

# Proportion of variance explained for each components
print(pca.explained_variance_ratio_)

plt.figure(figsize=(4, 4))

lw = 2

for color, target_name in zip(colors, types):
    plt.scatter(X_tr_pca[y_train == target_name, 0], X_tr_pca[y_train == target_name, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel('PC1 (69%)')
plt.ylabel('PC2 (20%)')
#plt.title('PCA of the Penguins dataset')
plt.show()
