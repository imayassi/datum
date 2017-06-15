from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, NMF
from sklearn.svm import SVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, mutual_info_regression,VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline



depth_panelty=0.005
def feature_clustering(df_no_pca,response, i):


    Y = df_no_pca[response]
    x=df_no_pca.drop([response], axis=1).astype(int)
    sel = VarianceThreshold(threshold=(.001 * (1 - .001)))
    new_x = sel.fit_transform(x)
    print 'new x ', new_x
    reduced_df = pd.DataFrame(new_x)
    sig_features = list(set(reduced_df.idxmax(axis=1).values))
    print  'sig features ', sig_features
    x2= df_no_pca[sig_features]
    print list(x2)
    y2 = df_no_pca[response]
    bool = pd.DataFrame(Y, columns=[response])
    df2 = pd.concat([df_no_pca[sig_features], bool[response]], axis=1)

    df=df2.sample(frac=0.1)
    print df.shape
    y = df[response]
    x = df.drop([response], axis=1)
    rs=np.random.RandomState(0)
    scores = ['precision']
    pipe = Pipeline([('reduce_dim', PCA(random_state=rs)), ('classify', LinearSVC())])
    N_FEATURES_OPTIONS = [30,60, 70, 80, 90, 100]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid =[
        {'reduce_dim': [PCA(), NMF()], 'reduce_dim__n_components': N_FEATURES_OPTIONS},
        {'reduce_dim': [SelectKBest(chi2)],'reduce_dim__k': N_FEATURES_OPTIONS,'classify__C': C_OPTIONS}
                ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

    grid = GridSearchCV(pipe, cv=3, n_jobs=2, param_grid=param_grid, scoring='precision')
    grid.fit(x,y)
    print("Best parameters set found on development set:")
    print(grid.best_params_['reduce_dim__k'])
    j=grid.best_params_['reduce_dim__k']

    # dict = {}
    # for m in range(50, 100, 10):
    #     # plsca = PLSRegression(n_components=m)
    #     # plsca.fit(x, y)
    #     # score = cross_val_score(plsca, x, y, scoring='r2')
    #     pca = PCA(n_components=m, random_state=np.random.RandomState(0))
    #     pca.fit_transform(x)
    #     x=pd.DataFrame(x)
    #     # svr=DecisionTreeRegressor(max_depth=1000)
    #     svr = DecisionTreeClassifier(max_depth=1000)
    #     svr.fit(x,y)
    #     score = cross_val_score(svr, x, y, scoring='precision')
    #     avg = np.mean(score) * 100
    #     dict[m] = avg - (m * depth_panelty)
    # print dict
    # j = max(dict.iterkeys(), key=lambda k: dict[k])
    # print j
    fc=i
    if fc == 'pls':
        # PLS appraoch
        print 'PLS appraoch'
        plsca = PLSRegression(n_components=j)
        plsca.fit(x2, y2)
        x3 = plsca.transform(x2)
        string = "pls_"
        pls_column_name = [string + `i` for i in range(x3.shape[1])]
        df1 = pd.DataFrame(x3, columns=pls_column_name)
        plsca_df = pd.DataFrame(plsca.x_weights_)
        plsca_trans = plsca_df.transpose()
        # x.reset_index(['CUSTOMER_KEY'], inplace=True)
        # x.drop(['CUSTOMER_KEY'], axis=1, inplace=True)
        reduced_df = pd.DataFrame(plsca_trans.values, columns=x2.columns, index=pls_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))

        x = df_no_pca
        df_final = x[sig_features]
        bool = pd.DataFrame(Y, columns=[response])
        # bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)

        df = pd.concat([df_final, bool[response]], axis=1)
        # df.set_index('CUSTOMER_KEY', inplace=True)

    #     PCA Approach
    elif fc == 'pca':
        pca = PCA(n_components=j, random_state=np.random.RandomState(0))
        pca.fit(x2)
        x3 = pca.transform(x2)
        string = "pca_"
        pca_column_name = [string + `i` for i in range(x3.shape[1])]
        reduced_df = pd.DataFrame(pca.components_, columns=x2.columns, index=pca_column_name)
        sig_features = list(set(reduced_df.idxmax(axis=1).values))

        x = df_no_pca
        df_final = x[sig_features]

        pca_df = reduced_df[sig_features]
        bool = pd.DataFrame(Y, columns=[response])
        # df_final.reset_index(level=['CUSTOMER_KEY'], inplace=True)
        df = pd.concat([df_final, bool[response]], axis=1)

        # df.set_index('CUSTOMER_KEY', inplace=True)

        plsca=pca

    elif fc=='none':

        bool = df2[response]
        # bool.reset_index(level=['CUSTOMER_KEY'], inplace=True)
        df = df2
        # df.set_index('CUSTOMER_KEY', inplace=True)
        print df.shape
        plsca = []

    return df, bool, plsca