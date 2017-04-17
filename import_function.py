import pyodbc

import pandas as pd
from sklearn import preprocessing
conn = pyodbc.connect(dsn='VerticaProd')
# from tto_pricing_features import tto_pricing_features

def import_scoring_data(scoring_data, cont_score_features, bool_score_features, catag_score_features):
    df = pd.read_sql(scoring_data, conn, index_col='CUSTOMER_KEY', coerce_float=False)
    df_cont = df[cont_score_features]


    df_cont.columns = df_cont.columns.str.strip()
    df_cont.fillna(value=0, inplace=True)
    df_cont.replace(to_replace=('(null)', 'NA', 'None'), value=0.000000001)
    df_cont = df_cont.astype(float)


    df_bool = df_cont[bool_score_features]
    df_bool = df_cont[bool_score_features]
    for f in df_bool.columns:
        if len(df_bool[f].unique()) < 2:
            df_bool.drop([f], axis=1, inplace=True)

    bool = df_bool.astype('bool')
    df_bool = bool
    df_cont.drop(bool_score_features, axis=1, inplace=True)

    index_df = pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])

    data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    data_scaled = pd.concat([data_scaled, index_df], axis=1)
    data_scaled.set_index('CUSTOMER_KEY', inplace=True)

    df_char = df[catag_score_features]
    df_char = df_char.astype(object)

    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA', 'None', '', ' ', '\t'), value='-1')

    just_dummies = pd.get_dummies(df_char).astype('bool')

    df_trans = pd.concat([df_bool, just_dummies, df_cont], axis=1)
    return df_trans


def import_data(data, cont_features, bool_features,response, catag_features):
    df = pd.read_sql(data, conn, coerce_float=False)

    df_cont = df[cont_features]
    df_cont.columns = df_cont.columns.str.strip()

    df_cont.fillna(value=0, inplace=True)
    df_cont.replace(to_replace=('(null)', 'NA'), value=0.000000001)


    df_cont = df_cont.astype(float)

    df_bool = df_cont[bool_features]
    df_bool['paid_something'] = 0

    df_bool['paid_something'][df[response] > 0]=1
    df_cont.drop([response], axis=1, inplace=True)
    df_bool['paid_something'].astype('bool')
    print df_bool
    for f in df_bool.columns:
        if len(df_bool[f].unique())<2:
            df_bool.drop([f], axis=1, inplace=True)

    bool=df_bool.astype('bool')
    df_bool=bool

    df_cont.drop(bool_features, axis=1, inplace=True)

    index_df = pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])
    print 'df_cont done'
    data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    print 'data_scaled done'

    df_char = df[catag_features]
    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies = pd.get_dummies(df_char).astype('bool')

    print 'just_dummies done'
    df_trans = pd.concat([df_bool, just_dummies, df_cont], axis=1)
    # df_trans.drop(['ABANDONED'], axis=1, inplace=True)
    print 'df_trans done'
    # new_list = list(set(list(df_trans)) & set(list(scoring_df)))
    print 'new feature list done'
    # df_trans_pca2 = df_trans[new_list]


    return df_trans





