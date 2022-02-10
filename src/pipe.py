import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif # classification
from sklearn.feature_selection import f_regression, mutual_info_regression # regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

# https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection
# https://scikit-learn.org/stable/modules/classes.html?highlight=selection%20feature#module-sklearn.feature_selection

with open('../data/STS/kinetic/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

X = data[:,0:56]  # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57] # rhs[0:50]

est = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVR())),
  ('regression', RandomForestRegressor())
])
est.fit(X, y.ravel())
