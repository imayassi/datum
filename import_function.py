import pyodbc

import pandas as pd
from sklearn import preprocessing
conn = pyodbc.connect(dsn='VerticaProd')


def import_scoring_data(scoring_data, cont_score_features, bool_score_features, catag_score_features):
    df = pd.read_sql(scoring_data, conn, index_col='CUSTOMER_KEY', coerce_float=False)
    # df = df_raw[df_raw['TAX_YEAR'] == 2016]
    print df

    df_cont = df[cont_score_features]
    # print df_cont

    df_cont.columns = df_cont.columns.str.strip()
    df_cont.fillna(value=0, inplace=True)
    df_cont.replace(to_replace=('(null)', 'NA', 'None'), value=0)
    df_cont = df_cont.astype(float)

    df_bool = df_cont[bool_score_features]
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

    # print list(df_char)
    # print df_char
    #
    just_dummies = pd.get_dummies(df_char)
    # print 'SCORING...', just_dummies
    df_trans = pd.concat([df_bool, just_dummies, data_scaled], axis=1)

    return df_trans, df_cont


def import_data(data, cont_features, bool_features, catag_features, scoring_df):
    # query = "SELECT * FROM CTG_ANALYTICS_WS.SM_TXML_TY13_TY14_S where  CUSTOMER_DEFINITION_ADJ IN ('NEW TO TURBOTAX')  ORDER BY RANDOM() LIMIT 5000"
    df = pd.read_sql(data, conn, index_col='CUSTOMER_KEY', coerce_float=False)
    # df = df_raw[df_raw['TAX_YEAR'] != 2016]

    df_cont = df[cont_features]
    df_cont.columns = df_cont.columns.str.strip()
    # print df_cont
    df_cont.fillna(value=0, inplace=True)
    df_cont.replace(to_replace=('(null)', 'NA'), value=0)

    df_cont = df_cont.astype(float)

    df_bool = df_cont[bool_features]
    df_cont.drop(bool_features, axis=1, inplace=True)

    index_df = pd.DataFrame(df_cont.reset_index(level=['CUSTOMER_KEY']), columns=['CUSTOMER_KEY'])
    print 'df_cont done'
    data_scaled = pd.DataFrame(preprocessing.normalize(df_cont), columns=df_cont.columns)
    data_scaled = pd.concat([data_scaled, index_df], axis=1)
    data_scaled.set_index('CUSTOMER_KEY', inplace=True)
    print 'data_scaled done'

    df_char = df[catag_features]
    df_char.columns = df_char.columns.str.strip()
    df_char.fillna(value='-1', inplace=True)
    df_char.replace(to_replace=('(null)', 'NA'), value='-1')
    just_dummies = pd.get_dummies(df_char)
    print 'just_dummies done'
    df_trans = pd.concat([df_bool, just_dummies, data_scaled], axis=1)
    print 'df_trans done'
    new_list = list(set(list(df_trans)) & set(list(scoring_df)))
    print 'new feature list done'
    df_trans_pca2 = df_trans[new_list]
    print list(df_trans_pca2)
    return df_trans_pca2, df_bool, df_cont



b=list(set(list(df_no_pca)) - set(list(scoring_df)))
print b
scoring_df=scoring_df[b]
print list(scoring_df)
