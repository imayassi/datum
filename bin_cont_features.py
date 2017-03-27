import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF
from binning import bin

def bin_pca(df_pca, bool_df, df_cont, b_pca):
    if b_pca=='True':
        pca_leng = {}

        # df_pca.drop(['ABANDONED'], inplace=True,axis=1, errors='ignore')
        print df_cont.dtypes
        lists=list(df_cont)
        print lists

        for k in lists:

            bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
            # print bool
            # bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)

            df_bin = pd.concat([df_cont[k], bool['ABANDONED']], axis=1)


            dict = bin(df_bin)

            leng = len(dict)

            pca_level = 'pcl_'
            labels = [pca_level + `r` for r in range(leng)]

            df_pca[k]= pd.cut(df_cont[k], bins=leng, labels=labels, include_lowest=True)
            print df_cont[k]
            pca_leng[k] = leng

        df_trans_pca_dummy = pd.get_dummies(df_cont)
        bool = pd.DataFrame(bool_df['ABANDONED'], columns=['ABANDONED'])
        bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
        df_pca.drop(lists, axis=1, inplace=True)
        training_df = pd.concat([df_trans_pca_dummy, bool['ABANDONED'],df_pca], axis=1)


    else:
        df_no_pca=df_pca
        training_df=pd.concat([df_no_pca, bool_df['ABANDONED']], axis=1)
        pca_leng={}

    return   training_df, pca_leng

