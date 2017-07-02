import pandas as pd
import numpy as np
import pickle
def transform_to_pca(data, fitted_pca, i):
    do_pca = i
    if do_pca == 'True':
        scoring_data = fitted_pca.transform(data)
        string = "pca_"
        pca_column_name = [string + `i` for i in range(scoring_data.shape[1])]
        df = pd.DataFrame(scoring_data, columns=pca_column_name)
        data.reset_index(['AUTH_ID'], inplace=True)

        data.drop(['AUTH_ID'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(fitted_pca.components_, columns=data.columns, index=pca_column_name)

        sig_features = list(set(reduced_df.idxmax(axis=1).values))
        df = data[sig_features]

    else:
        df = data


    return df


def bin_pca_score_set(df_trans_pca, length_dict, j):
    b_pca = j
    if b_pca == 'True':
        lists=list(df_trans_pca.select_dtypes(exclude=[np.bool]))
        df_cont=df_trans_pca.select_dtypes(exclude=[np.bool])
        for i in lists:
            if i in length_dict and length_dict[i]>2:
                print i
                pca_level = 'pcl_'

                dict_list = [length_dict[i] for i in length_dict]
                for m in dict_list:

                    df_trans_pca[i + m[0] + repr(m[1])] = 0
                    if m[0] == '<=' and len(m) == 2:
                        df_trans_pca[i + m[0] + repr(m[1])][df_trans_pca[i] <= m[1]] = 1

                    elif m[0] == '>=' and len(m) == 2:
                        df_trans_pca[i + m[0] + repr(m[1])][df_trans_pca[i] >= m[1]] = 1

                    elif len(m) == 4:
                        df_trans_pca[i + m[0] + repr(m[1]) + m[2] + repr(m[3])] = 0
                        df_trans_pca[i + m[0] + repr(m[1]) + m[2] + repr(m[3])][(df_trans_pca[i] > m[1]) & (df_trans_pca[i] < m[3])] = 1
                        df_trans_pca.drop([i], axis=1, inplace=True)

            else:
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
    x=scoring_df_trans
    if fc == 'pls':
        x3 = plsca.transform(x).values
        string = "pls_"
        pls_column_name = [string + `i` for i in range(x3.shape[1])]
        df1 = pd.DataFrame(x3, columns=pls_column_name)
        plsca_df = pd.DataFrame(plsca.x_weights_)
        plsca_trans = plsca_df.transpose()

        x.reset_index(['AUTH_ID'], inplace=True)
        index = x['AUTH_ID']
        x.drop(['AUTH_ID'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(plsca_trans.values, columns=x.columns, index=pls_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))
        df_final = pd.concat([x[sig_features], index], axis=1)
        df_final.set_index('AUTH_ID', inplace=True)
    if fc == 'pca':
        df_final = scoring_df_trans
        scoring_df_trans.reset_index(['AUTH_ID'], inplace=True)
        index = scoring_df_trans['AUTH_ID']


    return df_final, index


# name != "Support Vector" and name != "rbf_svc" and name != "poly_svc" and name != "lin_svc"
def predict(models, name, x_score, index):
    ABANDONED = {}
    for name, clf in zip(name, models):
        if name== "logistic_regression" :
            flag = clf.predict(x_score)
            likelihood = clf.predict_proba(x_score)
            defect_prob = [item[0] for item in likelihood]
            retain_prob = [item[1] for item in likelihood]
            ABANDONED[name + '_flag'] = flag
            ABANDONED[name + '_retain_prob'] = defect_prob
            ABANDONED[name + '_defect_prob'] = retain_prob
        else:
            flag = clf.predict(x_score)
            defect_prob = 0
            retain_prob = 0
            ABANDONED[name + '_flag'] = flag
            ABANDONED[name + '_retain_prob'] = defect_prob
            ABANDONED[name + '_defect_prob'] = retain_prob
    scored_df = pd.DataFrame.from_dict(ABANDONED)
    scored_df = pd.concat([scored_df, index], axis=1)
    scored_df.set_index('AUTH_ID', inplace=True)

    return scored_df