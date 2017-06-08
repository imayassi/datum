import pyodbc

import pandas as pd
from sklearn import preprocessing
import numpy as np
conn = pyodbc.connect(dsn='VerticaProd')
# from tto_pricing_features import tto_pricing_features

def import_scoring_data(scoring_data,care_score_data,scoring_data_PY, scoring_data_PY2, cont_score_features, bool_score_features, catag_score_features, care_cont_score_features, care_bool_score_features, care_catag_score_features):
    df = pd.read_sql(scoring_data, conn, index_col='AUTH_ID', coerce_float=False)
    df2 = pd.read_sql(care_score_data, conn, index_col='AUTH_ID', coerce_float=False)
    df3 = pd.read_sql(scoring_data_PY, conn, index_col='AUTH_ID', coerce_float=False)
    df4 = pd.read_sql(scoring_data_PY2, conn, index_col='AUTH_ID', coerce_float=False)
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

    df_trans = pd.concat([df_bool, just_dummies, df_cont], axis=1)
    # **************************************************************
    # ***************************importing base PY df***************
    # **************************************************************
    df_cont_py = df3[cont_score_features]
    df_cont_py.columns = df_cont_py.columns.str.strip()
    df_cont_py.fillna(value=0, inplace=True)
    med = df_cont_py.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py:
            df_cont_py[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py = df_cont_py.astype(float)
    df_bool_py = df_cont_py[bool_score_features]
    bool = df_bool_py.astype('bool')

    df_bool_py = bool

    df_cont_py.drop(bool_score_features, axis=1, inplace=True)
    print 'df_cont_py done'
    # data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    df_char_py = df3[catag_score_features]
    df_char_py.columns = df_char_py.columns.str.strip()
    df_char_py.fillna(value='-1', inplace=True)
    df_char_py.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py = pd.get_dummies(df_char_py).astype('bool')
    print 'just_dummies_py done'
    df_trans_py = pd.concat([df_bool_py, just_dummies_py, df_cont_py], axis=1)
    print 'df_trans_py done'

    # **************************************************************
    # ***************************importing base PY2 df***************
    # **************************************************************
    # print df4
    df_cont_py2 = df4[cont_score_features]
    df_cont_py2.columns = df_cont_py2.columns.str.strip()
    df_cont_py2.fillna(value=0, inplace=True)
    med = df_cont_py2.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py2:
            df_cont_py2[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py2 = df_cont_py2.astype(float)
    df_bool_py2 = df_cont_py2[bool_score_features]
    bool = df_bool_py2.astype('bool')
    df_bool_py2 = bool

    df_cont_py2.drop(bool_score_features, axis=1, inplace=True)
    print 'df_cont_py2 done'
    # data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    df_char_py2 = df4[catag_score_features]
    df_char_py2.columns = df_char_py2.columns.str.strip()
    df_char_py2.fillna(value='-1', inplace=True)
    df_char_py2.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py2 = pd.get_dummies(df_char_py2).astype('bool')
    print 'just_dummies_py2 done'
    df_trans_py2 = pd.concat([df_bool_py2, just_dummies_py2, df_cont_py2], axis=1)
    print 'df_trans_py2 done'

    # **************************************************************
    # ******joining df_trans & df_trans_py & df_trans_py2**********
    # **************************************************************
    boolean_df = pd.concat([df_bool, just_dummies], axis=1)
    boolean_df_py = pd.concat([df_bool_py, just_dummies_py], axis=1)
    boolean_df_py2 = pd.concat([df_bool_py2, just_dummies_py2], axis=1)
    df_j1 = boolean_df.join(boolean_df_py, rsuffix='_py').astype('bool').fillna(value='False')
    df_j2 = df_cont.join(df_cont_py, rsuffix='_py').astype(float).fillna(value=0)
    df_j3 = df_j1.join(boolean_df_py2, rsuffix='_py2').astype('bool').fillna(value='False')
    df_j4 = df_j2.join(df_cont_py2, rsuffix='_py2').astype(float).fillna(value=0)
    df_trans = pd.concat([df_j3, df_j4], axis=1)
    df_trans.fillna(value='False')
    # df_trans=df_j1.join(df_trans_py2,rsuffix='_py2')

    for column in df_trans.columns:
        if df_trans[column].dtype == np.bool:
            df_trans[column] = df_trans[column].fillna(value='False')
        else:
            df_trans[column] = df_trans[column].fillna(df_trans[column].median())



    # df_trans.drop(['ANC_PY_py','ANC_PY_py2' ], axis=1, inplace=True)
    # print df_trans.dtypes
    # **************************************************************
    # ***************************importing care df******************
    # **************************************************************
    df_cont_care = df2[care_cont_score_features]
    df_cont_care.columns = df_cont_care.columns.str.strip()

    df_bool_care = df_cont_care[care_bool_score_features]
    df_cont_care.fillna(value=0, inplace=True)

    df_cont_care.drop(care_bool_score_features, axis=1, inplace=True)
    med = df_cont_care.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_care:
            df_cont_care[j].replace(to_replace=('(null)', 'NA'), value=a[i])
        df_cont_care = df_cont_care.astype(float)

    df_cont_care.reset_index(['AUTH_ID'], inplace=True)
    df_cont_care.drop_duplicates(inplace=True)
    df_cont_care.set_index('AUTH_ID', inplace=True)
    cont = df_cont_care.groupby(level=0).max()
    # df_bool_care.drop_duplicates(inplace=True)
    # df_cont_care.drop_duplicates(inplace=True)
    bool = df_bool_care
    # .astype('bool')
    df_bool_care = bool
    df_bool_care.fillna(value=0, inplace=True)

    print 'df_cont done'

    data_scaled = pd.DataFrame(preprocessing.normalize(df_cont_care), columns=df_cont_care.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    print 'data_scaled done'

    df_char_care = df2[care_catag_score_features]
    df_char_care.columns = df_char_care.columns.str.strip()
    df_char_care.fillna(value='-1', inplace=True)
    df_char_care.replace(to_replace=('(null)', 'NA', 'NaN'), value='0')
    just_dummies_care = pd.get_dummies(df_char_care)
    just_dummies_care.fillna(value='0', inplace=True)
    # s=just_dummies_care.astype('bool')
    df_trans_care = pd.concat([df_bool_care, just_dummies_care], axis=1)

    df_trans_care.reset_index(['AUTH_ID'], inplace=True)
    df_trans_care.drop_duplicates(inplace=True)
    df_trans_care.set_index('AUTH_ID', inplace=True)
    s = df_trans_care.groupby(level=0).max().astype('bool')
    df_j6 = df_j3.join(s).astype('bool').fillna(value='False')
    df_j7 = df_j4.join(cont).astype(float).fillna(value=0)
    full_df = pd.concat([df_j6, df_j7], axis=1)
    # full_df.reset_index(['AUTH_ID'], inplace=True)
    # full_df.drop(['AUTH_ID'], axis=1, inplace=True)
    print full_df
    return full_df


def import_data(data,care_data,data_PY,data_PY2, cont_features, bool_features,response, catag_features,care_cont_features, care_bool_features,care_catag_features):
    df = pd.read_sql(data, conn,index_col='AUTH_ID', coerce_float=False)
    df2= pd.read_sql(care_data, conn,index_col='AUTH_ID', coerce_float=False)
    df3 = pd.read_sql(data_PY, conn, index_col='AUTH_ID', coerce_float=False)
    df4 = pd.read_sql(data_PY2, conn, index_col='AUTH_ID', coerce_float=False)
                    # **************************************************************
                    # ***************************importing base data df*************
                    # **************************************************************
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
    # data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    df_char = df[catag_features]
    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA'), value='-1')
    # print df_char
    just_dummies = pd.get_dummies(df_char).astype('bool')
    print 'just_dummies done'
    df_trans = pd.concat([df_bool, just_dummies, df_cont], axis=1)
    print 'df_trans done'
    # **************************************************************
    # ***************************importing base PY df***************
    # **************************************************************
    df_cont_py = df3[cont_features]
    df_cont_py.columns = df_cont_py.columns.str.strip()
    df_cont_py.fillna(value=0, inplace=True)
    med = df_cont_py.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py:
            df_cont_py[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py = df_cont_py.astype(float)
    df_bool_py = df_cont_py[bool_features]
    bool = df_bool_py.astype('bool')

    df_bool_py = bool

    df_cont_py.drop(bool_features, axis=1, inplace=True)
    print 'df_cont_py done'
    # data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    df_char_py = df3[catag_features]
    df_char_py.columns = df_char_py.columns.str.strip()
    df_char_py.fillna(value='-1', inplace=True)
    df_char_py.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py = pd.get_dummies(df_char_py).astype('bool')
    print 'just_dummies_py done'
    df_trans_py = pd.concat([df_bool_py, just_dummies_py, df_cont_py], axis=1)
    print 'df_trans_py done'




    # **************************************************************
    # ***************************importing base PY2 df***************
    # **************************************************************
    # print df4
    df_cont_py2 = df4[cont_features]
    df_cont_py2.columns = df_cont_py2.columns.str.strip()
    df_cont_py2.fillna(value=0, inplace=True)
    med = df_cont_py2.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_py2:
            df_cont_py2[j].replace(to_replace=('(null)', 'NA'), value=a[i])
    df_cont_py2 = df_cont_py2.astype(float)
    df_bool_py2 = df_cont_py2[bool_features]
    bool = df_bool_py2.astype('bool')
    df_bool_py2 = bool

    df_cont_py2.drop(bool_features, axis=1, inplace=True)
    print 'df_cont_py2 done'
    # data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    df_char_py2 = df4[catag_features]
    df_char_py2.columns = df_char_py2.columns.str.strip()
    df_char_py2.fillna(value='-1', inplace=True)
    df_char_py2.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies_py2 = pd.get_dummies(df_char_py2).astype('bool')
    print 'just_dummies_py2 done'
    df_trans_py2 = pd.concat([df_bool_py2, just_dummies_py2, df_cont_py2], axis=1)
    print 'df_trans_py2 done'



    # **************************************************************
    # ******joining df_trans & df_trans_py & df_trans_py2**********
    # **************************************************************
    boolean_df=pd.concat([df_bool, just_dummies], axis=1)
    boolean_df_py = pd.concat([df_bool_py, just_dummies_py], axis=1)
    boolean_df_py2 = pd.concat([df_bool_py2, just_dummies_py2], axis=1)
    df_j1=boolean_df.join(boolean_df_py,rsuffix='_py').astype('bool').fillna(value='False')
    df_j2=df_cont.join(df_cont_py, rsuffix='_py').astype(float).fillna(value=0)
    df_j3 = df_j1.join(boolean_df_py2, rsuffix='_py2').astype('bool').fillna(value='False')
    df_j4 = df_j2.join(df_cont_py2, rsuffix='_py2').astype(float).fillna(value=0)
    df_trans=pd.concat([df_j3,df_j4], axis=1)
    df_trans.fillna(value='False')
    # df_trans=df_j1.join(df_trans_py2,rsuffix='_py2')

    for column in df_trans.columns:
        if df_trans[column].dtype == np.bool:
            df_trans[column] = df_trans[column].fillna(value='False')
        else:
            df_trans[column] = df_trans[column].fillna(df_trans[column].median())

    df_trans.drop(['ABANDONED_py','ABANDONED_py2'], axis=1, inplace=True)

    # df_trans.drop(['ANC_PY_py','ANC_PY_py2' ], axis=1, inplace=True)
    # print df_trans.dtypes


    # **************************************************************
    # ***************************importing care df******************
    # **************************************************************
    df_cont_care = df2[care_cont_features]
    df_cont_care.columns = df_cont_care.columns.str.strip()

    df_bool_care = df_cont_care[care_bool_features]
    df_cont_care.fillna(value=0, inplace=True)

    df_cont_care.drop(care_bool_features, axis=1, inplace=True)
    med = df_cont_care.median(axis=0, skipna=True)
    a = list(med)
    for i in range(len(a)):
        for j in df_cont_care:
            df_cont_care[j].replace(to_replace=('(null)', 'NA'), value=a[i])
        df_cont_care = df_cont_care.astype(float)

    df_cont_care.reset_index(['AUTH_ID'], inplace=True)
    df_cont_care.drop_duplicates(inplace=True)
    df_cont_care.set_index('AUTH_ID', inplace=True)
    cont = df_cont_care.groupby(level=0).max()
    # df_bool_care.drop_duplicates(inplace=True)
    # df_cont_care.drop_duplicates(inplace=True)
    bool = df_bool_care
        # .astype('bool')
    df_bool_care = bool
    df_bool_care.fillna(value=0, inplace=True)

    print 'df_cont done'

    data_scaled = pd.DataFrame(preprocessing.normalize(df_cont_care), columns=df_cont_care.columns)
    # data_scaled = pd.concat([data_scaled, index_df], axis=1)
    # data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    print 'data_scaled done'

    df_char_care = df2[care_catag_features]
    df_char_care.columns = df_char_care.columns.str.strip()
    df_char_care.fillna(value='-1', inplace=True)
    df_char_care.replace(to_replace=('(null)', 'NA', 'NaN'), value='0')
    just_dummies_care = pd.get_dummies(df_char_care)
    just_dummies_care.fillna(value='0', inplace=True)
    # s=just_dummies_care.astype('bool')
    df_trans_care = pd.concat([df_bool_care, just_dummies_care], axis=1)

    df_trans_care.reset_index(['AUTH_ID'], inplace=True)
    df_trans_care.drop_duplicates(inplace=True)
    df_trans_care.set_index('AUTH_ID', inplace=True)
    s=df_trans_care.groupby(level=0).max().astype('bool')
    df_j6 = df_j3.join(s).astype('bool').fillna(value='False')
    df_j7=df_j4.join(cont).astype(float).fillna(value=0)
    full_df=pd.concat([df_j6,df_j7], axis=1)
    full_df.reset_index(['AUTH_ID'], inplace=True)
    full_df.drop(['AUTH_ID'],axis=1, inplace=True)

    print full_df


    return full_df





