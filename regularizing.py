import pandas as pd
import numpy as np
from tto_pricing_features import tto_pricing_features
ct='New'
response='TOTAL_REVENUE_CY'
cont_features, bool_features, catag_features, cont_score_features, bool_score_features, catag_score_features=tto_pricing_features(ct)
import pyodbc
conn = pyodbc.connect(dsn='VerticaProd')


for i in cont_features:
    print (i)
    data = "SELECT count(%d) FROM CTG_ANALYTICS_WS.SM_PRICING_MODEL WHERE %d is not null" % i
    df = pd.read_sql(data, conn, coerce_float=False)
    print (df)


