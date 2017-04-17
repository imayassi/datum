import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF
from binning import bin

def bin_pca(df_pca,response, j):
    b_pca=j
    if b_pca=='True':
        pca_leng = {}
        # df_pca.reset_index(level=['CUSTOMER_KEY'], inplace=True)
        # index=df_pca['CUSTOMER_KEY']
        # df_pca.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        y=df_pca[response]
        df_pca.drop([response], axis=1, inplace=True)
        lists=list(df_pca.select_dtypes(exclude=[np.bool]))
        df_cont=df_pca.select_dtypes(exclude=[np.bool])

        for k in lists:
            df_bin =pd.concat([y,df_pca[k]], axis=1) # when abandoned is the response use this function

            dict = bin(df_bin, response)

            # dict = bin(df_bin.drop([df_pca[response]], axis=1, inplace=True), df_pca[response])

            leng = len(dict)


            if leng>2:
                dict_list = [dict[i] for i in dict]
                print dict_list
                pca_level = 'pcl_'
                labels = [pca_level + `r` for r in dict_list]
                # range(leng)
                df_pca[k]= pd.cut(df_cont[k], bins=leng, labels=labels, include_lowest=True)
                pca_leng[k] = leng
            else:
                df_pca.drop([k], axis=1, inplace=True)
        df = pd.get_dummies(df_pca).astype('bool')
        training_df=pd.concat([df, y], axis=1)
    else:
        training_df=df_pca
        pca_leng={}
    return   training_df, pca_leng

# dict={1:'a', 2:'b'}
# leng = len(dict)
# dict_list = [dict[i] for i in dict]
# print dict_list