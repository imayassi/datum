from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

depth_panelty=0.002
def feature_clustering(x, y,bool_df, i):
    dict = {}
    for m in range(50, 300, 50):
        plsca = PLSRegression(n_components=m)
        plsca.fit(x, y)
        score = cross_val_score(plsca, x, y, scoring='roc_auc')
        avg = np.mean(score) * 100
        dict[m] = avg - (m * depth_panelty)
    print dict
    j = max(dict.iterkeys(), key=lambda k: dict[k])
    print j
    fc=i
    if fc == 'True':
        print 'test failed'
        plsca = PLSRegression(n_components=j)
        plsca.fit(x, y)
        x3 = plsca.transform(x)
        string = "pls_"
        pls_column_name = [string + `i` for i in range(x3.shape[1])]
        df1 = pd.DataFrame(x3, columns=pls_column_name)
        plsca_df = pd.DataFrame(plsca.x_weights_)
        plsca_trans = plsca_df.transpose()
        x.reset_index(['CUSTOMER_KEY'], inplace=True)
        x.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(plsca_trans.values, columns=x.columns, index=pls_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))

        df_final = x[sig_features]
        bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
        bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
        df = pd.concat([df_final, bool['ABANDONED']], axis=1)

    else:
        df_final = x
        bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
        plsca = []

    return df_final, bool['ABANDONED'], plsca