import pyodbc

import pandas as pd
from sklearn import preprocessing
import numpy as np
conn = pyodbc.connect(dsn='VerticaProd')
# from tto_pricing_features import tto_pricing_features

def import_scoring_data(scoring_data,care_score_data, cont_score_features, bool_score_features, catag_score_features, care_cont_score_features, care_bool_score_features, care_catag_score_features):
    df = pd.read_sql(scoring_data, conn, index_col='AUTH_ID', coerce_float=False)
    df2 = pd.read_sql(care_score_data, conn, index_col='AUTH_ID', coerce_float=False)

    df_cont = df[cont_score_features]
    df_cont.columns = df_cont.columns.str.strip()
    df_cont.fillna(value=0, inplace=True)
    avg=df_cont.mean(axis=0, skipna=True)
    a=list(avg)
    for i in range(len(a)):
        for j in df_cont:
            df_cont[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont = df_cont.astype(float)
    df_bool = df_cont[bool_score_features]
    for f in df_bool.columns:
        if len(df_bool[f].unique()) < 2:
            df_bool.drop([f], axis=1, inplace=True)

    bool = df_bool.astype('bool')
    df_bool = bool
    df_cont.drop(bool_score_features, axis=1, inplace=True)

    index_df = pd.DataFrame(df_cont.reset_index(level=['AUTH_ID']), columns=['AUTH_ID'])

    data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    data_scaled = pd.concat([data_scaled, index_df], axis=1)
    data_scaled.set_index('AUTH_ID', inplace=True)

    df_char = df[catag_score_features]
    df_char = df_char.astype(object)

    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA', 'None', '', ' ', '\t'), value='-1')
    b=list(df_char)
    just_dummies = pd.get_dummies(df_char).astype('bool')

    df_j3 = pd.concat([df_bool, just_dummies, df_cont], axis=1)

    # **************************************************************
    # ***************************importing care df******************
    # **************************************************************
    # df_cont_care = df2[care_cont_score_features]
    # df_cont_care.columns = df_cont_care.columns.str.strip()
    #
    # df_bool_care = df_cont_care[care_bool_score_features]
    # df_cont_care.fillna(value=0, inplace=True)
    #
    # df_cont_care.drop(care_bool_score_features, axis=1, inplace=True)
    # med = df_cont_care.median(axis=0, skipna=True)
    # a = list(med)
    # for i in range(len(a)):
    #     for j in df_cont_care:
    #         df_cont_care[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    #     df_cont_care = df_cont_care.astype(float)
    #
    # df_cont_care.reset_index(['AUTH_ID'], inplace=True)
    # df_cont_care.drop_duplicates(inplace=True)
    # df_cont_care.set_index('AUTH_ID', inplace=True)
    # cont = df_cont_care.groupby(level=0).max()
    #
    # bool = df_bool_care
    # # .astype('bool')
    # df_bool_care = bool
    # df_bool_care.fillna(value=0, inplace=True)
    #
    # print 'df_cont done'
    #
    # print 'data_scaled done'
    #
    # df_char_care = df2[care_catag_score_features]
    # df_char_care.columns = df_char_care.columns.str.strip()
    # df_char_care.fillna(value='-1', inplace=True)
    # df_char_care.replace(to_replace=('(null)', 'NA', 'NaN'), value='0')
    # just_dummies_care = pd.get_dummies(df_char_care)
    # just_dummies_care.fillna(value='0', inplace=True)
    # # s=just_dummies_care.astype('bool')
    # df_trans_care = pd.concat([df_bool_care, just_dummies_care], axis=1)
    #
    # df_trans_care.reset_index(['AUTH_ID'], inplace=True)
    # df_trans_care.drop_duplicates(inplace=True)
    # df_trans_care.set_index('AUTH_ID', inplace=True)
    # s = df_trans_care.groupby(level=0).max().astype('bool')
    # df_j6 = df_j3.join(s).astype('bool').fillna(value='False')
    # df_j7 = df_cont.join(cont).astype(float).fillna(value=0)
    # full_df = pd.concat([df_j6, df_j7], axis=1)
    # print full_df.shape
    full_df=df_j3
    return full_df


def import_data(data,care_data, cont_features, bool_features, catag_features,care_cont_features, care_bool_features,care_catag_features):
    df = pd.read_sql(data, conn,index_col='AUTH_ID', coerce_float=False)
    df2 = pd.read_sql(care_data, conn, index_col='AUTH_ID', coerce_float=False)
    df_cont = df[cont_features]
    df_cont.columns = df_cont.columns.str.strip()
    df_cont.fillna(value=0, inplace=True)
    med=df_cont.median(axis=0, skipna=True)
    a=list(med)
    for i in range(len(a)):
        for j in df_cont:
            df_cont[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont = df_cont.astype(float)
    df_bool = df_cont[bool_features]
    bool=df_bool.astype('bool')
    df_bool=bool

    df_cont.drop(bool_features, axis=1, inplace=True)
    print 'df_cont done'
    df_char = df[catag_features]
    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies = pd.get_dummies(df_char).astype('bool')
    print 'just_dummies done'
    df_j3 = pd.concat([df_bool, just_dummies, df_cont], axis=1)
    full_df=df_j3
    print 'df_trans done'


    # **************************************************************
    # ***************************importing care df******************
    # **************************************************************
    # df_cont_care = df2[care_cont_features]
    # df_cont_care.columns = df_cont_care.columns.str.strip()
    #
    # df_bool_care = df_cont_care[care_bool_features]
    # df_cont_care.fillna(value=0, inplace=True)
    #
    # df_cont_care.drop(care_bool_features, axis=1, inplace=True)
    # med = df_cont_care.median(axis=0, skipna=True)
    # a = list(med)
    # for i in range(len(a)):
    #     for j in df_cont_care:
    #         df_cont_care[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    #     df_cont_care = df_cont_care.astype(float)
    #
    # df_cont_care.reset_index(['AUTH_ID'], inplace=True)
    # df_cont_care.drop_duplicates(inplace=True)
    # df_cont_care.set_index('AUTH_ID', inplace=True)
    # cont = df_cont_care.groupby(level=0).max()
    #
    # bool = df_bool_care
    #     # .astype('bool')
    # df_bool_care = bool
    # df_bool_care.fillna(value=0, inplace=True)
    #
    # print 'df_cont done'
    #
    #
    # print 'data_scaled done'
    #
    # df_char_care = df2[care_catag_features]
    # df_char_care.columns = df_char_care.columns.str.strip()
    # df_char_care.fillna(value='-1', inplace=True)
    # df_char_care.replace(to_replace=('(null)', 'NA', 'NaN'), value='0')
    # just_dummies_care = pd.get_dummies(df_char_care)
    # just_dummies_care.fillna(value='0', inplace=True)
    # # s=just_dummies_care.astype('bool')
    # df_trans_care = pd.concat([df_bool_care, just_dummies_care], axis=1)
    #
    # df_trans_care.reset_index(['AUTH_ID'], inplace=True)
    # df_trans_care.drop_duplicates(inplace=True)
    # df_trans_care.set_index('AUTH_ID', inplace=True)
    # s=df_trans_care.groupby(level=0).max().astype('bool')
    # df_j6 = df_j3.join(s).astype('bool').fillna(value='False')
    # df_j7=df_cont.join(cont).astype(float).fillna(value=0)
    # full_df=pd.concat([df_j6,df_j7], axis=1)
    # full_df.reset_index(['AUTH_ID'], inplace=True)
    # full_df.drop(['AUTH_ID'],axis=1, inplace=True)
    #
    # print full_df


    return full_df





