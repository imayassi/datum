import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF

def pca_code(df_no_pca,df_cont,bool_df, i):
    do_pca=i
    if do_pca=='True':
        pca = PCA(n_components=100, random_state=np.random.RandomState(0))
        pca.fit(df_no_pca)
        x3 = pca.transform(df_no_pca)
        string = "pca_"
        pca_column_name=[string+`i` for i in range(x3.shape[1])]
        df1=pd.DataFrame(x3, columns=pca_column_name)
        df_no_pca.reset_index(['CUSTOMER_KEY'], inplace=True)
        df_no_pca.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(pca.components_, columns=df_no_pca.columns, index=pca_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))
        df_final = df_no_pca[sig_features]
        bool=pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
        bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
        df = pd.concat([df_final, bool['ABANDONED']], axis=1)

    else:
        df=pd.concat([df_no_pca, bool_df['ABANDONED']], axis=1)
        pca=()
    return df, pca, df_cont