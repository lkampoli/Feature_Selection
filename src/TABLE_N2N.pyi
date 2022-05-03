import pandas as pd
import matplotlib.pyplot as plt

# from pandas.table.plotting import tabl
df = pd.read_csv("C:/Users/user/Feature_Selection/data/STS/kinetic/corr_df_N2N.csv", sep=",", header=0,)

df11 = df.rename(columns={"index": "features", "RDm1": "Pearson Correlation"})
df12 = df11.iloc[:-1]
df1 = pd.read_csv("C:/Users/user/Feature_Selection/data/STS/kinetic/Lasso_N2N.csv", sep=",", header=0, index_col='index')
df1 = df1.reset_index().drop("index", axis=1)
new_DF = pd.concat([df12, df1], axis=1).set_index("features")
dm = pd.concat([new_DF,df3], axis=1)
df3 = pd.read_csv("C:/Users/user/Feature_Selection/data/STS/kinetic/SelectFromModel.csv", sep=",", header=0, index_col='index')
dm1 = dm.loc[('X48','SelectFromModel')] = 0

dRFE = pd.read_csv("C:/Users/user/Feature_Selection/data/STS/kinetic/RFE_N2N.csv", sep=",", header=0, index_col='index')
dRFE = dRFE.reset_index().drop("index", axis=1)
dm_new = pd.concat([dm,dRFE], axis=1)
dm["RFE"] = dRFE["RFE"].values
dRF = pd.read_csv("C:/Users/user/Feature_Selection/data/STS/kinetic/RF_importances_mea_N2N.csv", sep=",", header=0, index_col='index')

dRF = dRF.reset_index().drop("index", axis=1)
dm["importances_mea"] = dRF["importances_mea"].values
dm = dm.rename(columns={"index": "features", "importances_mea": "RF"})
dSeq = pd.read_csv("C:/Users/user/Feature_Selection/data/STS/kinetic/SequentialFeatureSelector_N2N.csv", sep=",", header=0, index_col='index')

dSeq = dSeq.reset_index().drop("index", axis=1)
dm["SequentialFeatureSelector"] = dSeq["SequentialFeatureSelector"].values

print(dm)





