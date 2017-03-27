import pandas as pd
from sklearn.decomposition import PCA, NMF
from binning import bin
import numpy as np
from statsmodels.tools import categorical
import skfuzzy as fuzz
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn import datasets, cluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score, roc_auc_score,  average_precision_score, f1_score
from sklearn.cross_decomposition import PLSCanonical
from sklearn import linear_model, decomposition, datasets
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from binning import bin
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from features_by_customer_type import customer_type_features

import pyodbc
import pandas as pd
random_state = np.random.RandomState(0)
ct='New'
cont_features, bool_features, catag_features, _, _, _=customer_type_features(ct)
# , 'PY SKIPPER - PAST SKIPPER','PY SKIPPER'
# 'NEW TO TURBOTAX'
conn = pyodbc.connect(dsn='VerticaProd')
random_state = np.random.RandomState(0)
data = "SELECT * FROM (SELECT * FROM  CTG_ANALYTICS_WS.SM_RETENTION_MODEL  WHERE   TAX_YEAR<2014 AND TAX_DAY<=150)A   order by random() limit 100000"
df = pd.read_sql(data, conn, index_col=['CUSTOMER_KEY'], coerce_float=False)
print list(df)
df_cont = df[cont_features]
df_cont.columns = df_cont.columns.str.strip()
df_cont.fillna(value=0, inplace=True)
df_cont.replace(to_replace=('(null)', 'NA', 'None'), value=0)
df_bool = df_cont[bool_features]
df_cont.drop(bool_features, axis=1, inplace=True)
index_df = pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])

data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
data_scaled = pd.concat([data_scaled, index_df], axis=1)
data_scaled.set_index('CUSTOMER_KEY', inplace=True)

df_char = df[catag_features]
df_char.columns = df_char.columns.str.strip()
df_char.fillna(value='-1', inplace=True)
df_char.replace(to_replace=('(null)', 'NA', 'None'), value='-1')
just_dummies = pd.get_dummies(df_char, prefix=catag_features)

df_trans = pd.concat([df_bool, just_dummies, data_scaled], axis=1)
df_trans.reset_index(['CUSTOMER_KEY'], inplace=True)

y=df_trans[['CUSTOMER_KEY','ABANDONED']]
y.set_index('CUSTOMER_KEY', inplace=True)
x=df_trans.drop(['ABANDONED'], axis=1)
x.set_index('CUSTOMER_KEY', inplace=True)
depth_panelty=0.002
# PLS Regression to reduce number of variables and avoid multicollinearity
dict={}
for i in range(50,300, 50):
    plsca = PLSRegression(n_components=i)
    plsca.fit(x, y)
    score = cross_val_score(plsca, x, y, scoring='roc_auc')
    avg=np.mean(score)*100
    dict[i]=avg-(i*depth_panelty)
print dict
j= max(dict.iterkeys(), key=lambda k: dict[k])
print j

pca = PCA(n_components=j, random_state=np.random.RandomState(0))
pca.fit(x)
x3 = pca.transform(x)
string = "pca_"
pca_column_name = [string + `i` for i in range(x3.shape[1])]
reduced_df=pd.DataFrame(pca.components_,columns=x.columns,index =pca_column_name)
sig_features=list(set(reduced_df.idxmax(axis=1).values))
print sig_features
df_final=x[sig_features]
pca_df=reduced_df[sig_features]





plsca = PLSRegression(n_components=j)
plsca.fit(x,y)
x_pls = plsca.transform(x)
string = "pls_"
x_pls_column_name = [string + `i` for i in range(x_pls.shape[1])]
plsca_df=pd.DataFrame(plsca.x_weights_)
plsca_trans=plsca_df.transpose()
x_pls_reduced_df=pd.DataFrame(plsca_trans.values,columns=x.columns,index =x_pls_column_name)
pls_sig_features=list(set(x_pls_reduced_df.idxmax(axis=1).values))
print pls_sig_features
df_trans.reset_index(['CUSTOMER_KEY'], inplace=True)
pls_final=pd.concat([df_trans[pls_sig_features], df_trans['CUSTOMER_KEY']], axis=1)
y.reset_index(['CUSTOMER_KEY'], inplace=True)
df2=pd.concat([y,pls_final], axis=1)
df2.set_index('CUSTOMER_KEY', inplace=True)


y=df2['ABANDONED']
x=df2.drop(['ABANDONED'], axis=1)
clf=RandomForestClassifier(criterion='entropy', n_estimators=100)
clf.fit(x,y)
names=list(x)
feature_df=pd.DataFrame(clf.feature_importances_,columns=['sig'], index=names).sort_values(['sig'], ascending=False)
print feature_df















