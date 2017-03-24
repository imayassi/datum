import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
def transform_to_pca(data, fitted_pca, i):
    do_pca = i
    if do_pca == 'True':
        scoring_data = fitted_pca.transform(data)
        string = "pca_"
        pca_column_name = [string + `i` for i in range(scoring_data.shape[1])]
        df = pd.DataFrame(scoring_data, columns=pca_column_name)
        data.reset_index(['CUSTOMER_KEY'], inplace=True)

        data.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(fitted_pca.components_, columns=data.columns, index=pca_column_name)

        sig_features = list(set(reduced_df.idxmax(axis=1).values))
        df = data[sig_features]

    else:
        df = data

    return df


def bin_pca_score_set(df_trans_pca, length_dict, j):
    b_pca = j
    if b_pca == 'True':
        pca_leng = {}
        leng = len(length_dict)
        for i in df_trans_pca.columns:
            if i in length_dict:
                pca_level = 'pcl_'
                labels = [pca_level + `r` for r in range(length_dict[i])]
                df_trans_pca[i] = pd.cut(df_trans_pca[i], bins=length_dict[i], labels=labels, include_lowest=True)
                pca_leng[i] = leng
            elif i not in length_dict:
                df_trans_pca.drop([i], axis=1, inplace=True)
        df_trans_pca_dummy = pd.get_dummies(df_trans_pca)
        scoring_df_trans = df_trans_pca_dummy

    else:
        scoring_df_trans = df_trans_pca

    return scoring_df_trans


def get_scoring_arrays(scoring_df_trans):
    x_score = scoring_df_trans
    return x_score


def score_set_feature_selection(scoring_df_trans, plsca, k):
    fc = k
    if fc == 'True':
        x3 = plsca.transform(x)
        string = "pls_"
        pls_column_name = [string + `i` for i in range(x3.shape[1])]
        df1 = pd.DataFrame(x3, columns=pls_column_name)
        plsca_df = pd.DataFrame(plsca.x_weights_)
        plsca_trans = plsca_df.transpose()

        x.reset_index(['CUSTOMER_KEY'], inplace=True)
        index = x['CUSTOMER_KEY']
        x.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(plsca_trans.values, columns=x.columns, index=pls_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))

        df_final = pd.concat([x[sig_features], index], axis=1)
        df_final.set_index('CUSTOMER_KEY', inplace=True)
    else:
        # x.set_index('CUSTOMER_KEY', inplace=True)
        df_final = scoring_df_trans
        scoring_df_trans.reset_index(['CUSTOMER_KEY'], inplace=True)
        index = scoring_df_trans['CUSTOMER_KEY']

    return df_final, index


def predict(models, name, x_score, index):
    ABANDONED = {}
    for name, clf in zip(name, models):
        flag = clf.predict(x_score)
        likelihood = clf.predict_proba(x_score)
        defect_prob = [item[0] for item in likelihood]
        retain_prob = [item[1] for item in likelihood]
        ABANDONED[name + '_flag'] = flag
        ABANDONED[name + '_retain_prob'] = defect_prob
        ABANDONED[name + '_defect_prob'] = retain_prob
    scored_df = pd.DataFrame.from_dict(ABANDONED)
    scored_df = pd.concat([scored_df, index], axis=1)
    scored_df.set_index('CUSTOMER_KEY', inplace=True)
    return scored_df