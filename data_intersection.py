import pandas as pd
import numpy as np
def scoring_data_intersection(df_no_pca, scoring_df):

        new_list2=list(set(list(df_no_pca)) & set(list(scoring_df)))
        df_scoring2=scoring_df[new_list2]
        print "intersection of dataframes is done"

        return df_scoring2