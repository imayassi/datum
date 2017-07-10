
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score,  average_precision_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from pandas import DataFrame
import pickle
random_state = np.random.RandomState(0)
from sklearn.utils import shuffle
from sklearn import linear_model
def algorithm(x,y, response):
    df=pd.concat([x,y], axis=1)
    y = df[response]
    x = df.drop(response, 1)
    models = []
    x, y = shuffle(x, y, random_state=np.random.RandomState(0))
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=np.random.RandomState(0))
    C=1.0
    names = [
        # "Nearest Neighbors" ,
        # "Support Vector",
        # "rbf_svc",
        # "poly_svc",
        # "lin_svc",
        "Decision_Tree",
        "Random_Forest",
        "logistic_regression"
        # "NeuralNetworkLogistic",
        # "NeuralNetwork",
        # # "AdaBoost",
        #  "Naive Bayes",
        # "Bernouli Niave Bayes"
        # # "QDA",
        # "Bagging" ,
        # "ERT",
        # "GB"
    ]

    classifiers = [
        # KNeighborsClassifier(n_neighbors=5, leaf_size=1),
        # svm.SVC(kernel='linear', C=C,class_weight= 'balanced',random_state=np.random.RandomState(0)),
        # svm.SVC(kernel='rbf', gamma=0.7, C=C, class_weight='balanced',random_state=np.random.RandomState(0)),
        # svm.SVC(kernel='poly', degree=3, C=C,  class_weight= 'balanced',random_state=np.random.RandomState(0)),
        # svm.LinearSVC(C=C,  class_weight= {0:.5, 1:.5},probability=True),
        DecisionTreeClassifier(criterion='entropy'),
        RandomForestClassifier(criterion='entropy', n_estimators=200, random_state=np.random.RandomState(0)),
        linear_model.LogisticRegression( random_state=np.random.RandomState(0))
        # MLPClassifier(alpha=1e-5,activation='logistic', random_state = random_state),
        # MLPClassifier(alpha=1e-5, random_state=random_state),
        # # AdaBoostClassifier(n_estimators=100),
        # GaussianNB(),
        # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        # QuadraticDiscriminantAnalysis(),
        # BaggingClassifier(bootstrap_features=True,random_state=np.random.RandomState(0)),
        # ExtraTreesClassifier(criterion='entropy', random_state=np.random.RandomState(0)),
        # GradientBoostingClassifier(n_estimators=1000, max_depth=10000, random_state= np.random.RandomState(0))
    ]




    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision = precision_score(y_test, y_pred)
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
            naming = list(X_train)
            feature_df = pd.DataFrame(clf.feature_importances_, columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            feature_df.to_csv(path_or_buf='defection_model_features_rf_ANC.txt', index=True)

        elif name == "logistic_regression":
            naming = list(X_train)
            feature_df = pd.DataFrame(clf.coef_[0], columns=['sig'], index=naming).abs().sort_values(['sig'],ascending=False)
            feature_df2 = pd.DataFrame(clf.coef_[0], columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            feature_df2.to_csv(path_or_buf='defection_model_features_ANC.txt', index=True)
            feature_df2.reset_index(['naming'], inplace=True)
            print feature_df2




            top_features=feature_df2[np.exp(feature_df2['sig']) >=1.1 ]

            print top_features

            top_df = pd.concat([x[top_features['index'].tolist()], y], axis=1)
            top_df2 = top_df.sample(frac=0.1)
            print 'Features with >1.1 odds ratio', list(top_df)
            y = top_df2[response]
            x = top_df2.drop(response, 1)

            x, y = shuffle(x, y, random_state=np.random.RandomState(0))
            y = y.astype(int)
            poly = PolynomialFeatures(3)
            r = poly.fit_transform(x)
            # print list(x)
            feature_interaction=poly.get_feature_names(list(x))
            df=DataFrame(r, columns=feature_interaction)



            X_train2, X_test2, y_train2, y_test2 = train_test_split(df, y, test_size=.3,random_state=np.random.RandomState(0))
            reg=linear_model.LogisticRegression()
            reg.fit(X_train2, y_train2)
            naming = list(X_train2)
            feature_df2 = pd.DataFrame(reg.coef_[0], columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            feature_df2.to_csv(path_or_buf='defection_model_segments_ANC.txt', index=True)
            # print feature_df2
            y_pred2 = reg.predict(X_test2)
            precision = average_precision_score(y_test2, y_pred2)
            recall = recall_score(y_test2, y_pred2)
            auc = roc_auc_score(y_test2, y_pred2)
            print precision, recall, auc




            # , "recall_avg ", recall_avg, "precision_avg ", precision_avg

        filename = 'finalized_model.sav'
        pickle.dump(clf, open(filename, 'wb'))
        filename2 = 'name.sav'
        pickle.dump(clf, open(filename2, 'wb'))

        models.append(clf)
        # return name, ' model', ' precision score', precision, ' recall score', recall, ' f1 ', f1




    return models, names

