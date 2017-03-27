import pandas as pd
import numpy as np
def get_arrays(dummy_pca,df_pca,df_no_pca, bool_df, i, j):
    do_pca=i
    b_pca=j
    response_feature = ['ABANDONED']
    if do_pca=='True' and b_pca=='True':
        Y = dummy_pca['ABANDONED']
        X = dummy_pca.drop(response_feature, 1)
        y = Y
        x = X

    elif do_pca=='True' and b_pca!='True':

        Y = df_pca['ABANDONED']
        X = df_pca.drop(response_feature, 1)
        y = Y
        x = X

    else:
        Y = bool_df['ABANDONED']
        X = df_no_pca
        y = Y
        x = X

    return x, y
