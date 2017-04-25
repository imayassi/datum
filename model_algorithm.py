
import pandas as pd
from sklearn.decomposition import PCA, NMF
from binning import bin
import numpy as np
from statsmodels.tools import categorical
import skfuzzy as fuzz
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn import datasets, cluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,VotingClassifier,RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score, roc_auc_score,  average_precision_score, f1_score
from sklearn.cross_decomposition import PLSCanonical
from sklearn import linear_model, decomposition, datasets
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from binning import bin
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
# import vertica_python
import pyodbc
import pandas as pd
from pandas import Series,DataFrame
# from retention_customer_type_features_new import customer_type_features
from features_by_customer_type import customer_type_features
from sklearn.feature_selection import SelectKBest, chi2
import pickle
random_state = np.random.RandomState(0)
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def algorithm(x,y, response):
    df=pd.concat([x,y], axis=1)
    label=list(df['LABELS'])
    # for i in label:
    #     print i
    y = df[response]
    x = df.drop(response, 1)
    x.drop(['LABELS'], axis=1, inplace=True)
    models = []
    x, y = shuffle(x, y, random_state=np.random.RandomState(0))
    y = y.astype(int)
    poly = PolynomialFeatures(2)
    # r=poly.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=np.random.RandomState(0))

    names = [
        # "Nearest Neighbors" ,
        # "Decision_Tree",
        "Random_Forest",
        "logistic_regression"
        # "NeuralNetworkLogistic",
        # "NeuralNetwork",
        # "AdaBoost",
        #  "Naive Bayes",
        # "Bernouli Niave Bayes",
        # "QDA",
        # "Bagging" ,
        # "ERT",
        # "GB"
    ]

    classifiers = [
        # KNeighborsClassifier(n_neighbors=5, leaf_size=1),
        # DecisionTreeClassifier(criterion='entropy'),
        RandomForestClassifier(criterion='entropy', n_estimators=200),
        linear_model.LogisticRegression()
        # MLPClassifier(alpha=1e-5,activation='logistic', random_state = random_state),
        # MLPClassifier(alpha=1e-5, random_state=random_state),
        # AdaBoostClassifier(n_estimators=100),
        # GaussianNB(),
        # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True),
        # QuadraticDiscriminantAnalysis(),
        # BaggingClassifier(bootstrap_features=True,random_state=np.random.RandomState(0)),
        # ExtraTreesClassifier(criterion='entropy', random_state=np.random.RandomState(0)),
        # GradientBoostingClassifier(n_estimators=1000, max_depth=10000, random_state= np.random.RandomState(0))
    ]
    # names = [
    #     # "Nearest Neighbors" ,
    #     "Decision_Tree",
    #     "RandomForestRegressor",
    #     "svr_rbf",
    #     "svr_lin",
    #     "svr_poly",
    #     "regression",
    #     "MLP_Regressor"
    # ]
    #
    # classifiers = [
    #     # KNeighborsClassifier(n_neighbors=20, leaf_size=1),
    #     DecisionTreeRegressor(max_depth=1000),
    #     RandomForestRegressor(n_estimators=200,random_state= np.random.RandomState(0)),
    #     SVR(kernel='rbf', C=1e3, gamma=0.1),
    #     SVR(kernel='linear', C=1e3),
    #     SVR(kernel='poly', C=1e3, degree=2),
    #     linear_model.LinearRegression(),
    #     MLPRegressor()
    # ]




    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision = average_precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc=roc_auc_score(y_test, y_pred)
        f1=f1_score(y_test, y_pred)
        tn, fp, fn, tp=confusion_matrix(y_test, y_pred).ravel()

        # r2=r2_score(y_test, y_pred)
        # mse=mean_squared_error(y_test, y_pred)


        print name , precision, recall, f1, auc, tn, fp, fn, tp
        # print name , ' r2=',r2, ' mse=',mse

        naming = list(X_train)
        if name=="Random_Forest":
            # print clf.feature_importances_
            feature_df = pd.DataFrame(clf.feature_importances_, columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            print feature_df
        elif name == "logistic_regression":
            # print list(clf.coef_),clf.coef_[0]
            feature_df = pd.DataFrame(clf.coef_[0], columns=['sig'], index=naming).abs().sort_values(['sig'],ascending=False)
            feature_df2 = pd.DataFrame(clf.coef_[0], columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            feature_df2.to_csv(path_or_buf='defection_model_features.txt', index=True)
            feature_df2.reset_index(['naming'], inplace=True)
            print feature_df2



        # filename = 'finalized_model.sav'
        # pickle.dump(clf, open(filename, 'wb'))
            print feature_df
            print feature_df2
            top_features = feature_df2.nlargest(10, 'sig')
            top_features.drop(['sig'], axis=1, inplace=True)
            print list(top_features)
            top_df = pd.concat([df[top_features['index'].tolist()], y], axis=1)

            y = top_df[response]
            x = top_df.drop(response, 1)

            x, y = shuffle(x, y, random_state=np.random.RandomState(0))
            y = y.astype(int)
            poly = PolynomialFeatures(3)
            r = poly.fit_transform(x)
            print list(x)
            feature_interaction=poly.get_feature_names(list(x))
            df=DataFrame(r, columns=feature_interaction)



            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=.3,random_state=np.random.RandomState(0))
            reg=linear_model.LogisticRegression()
            reg.fit(X_train, y_train)
            naming = list(X_train)
            feature_df2 = pd.DataFrame(reg.coef_[0], columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            feature_df2.to_csv(path_or_buf='defection_model_segments.txt', index=True)
            print feature_df2
            y_pred = reg.predict(X_test)
            precision = average_precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            print precision, recall, auc


            # , "recall_avg ", recall_avg, "precision_avg ", precision_avg

        models.append(clf)
        # return name, ' model', ' precision score', precision, ' recall score', recall, ' f1 ', f1




    return models, names

