
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
from sklearn.svm import SVC, LinearSVC
import pickle
random_state = np.random.RandomState(0)
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn import metrics
import itertools


scorer = 'roc_auc'
random_state=12345
def algorithm(x,y, response):
    df=pd.concat([x,y], axis=1)
    # y = df[response]
    # x = df.drop(response, 1)
    models = []
    names=[]
    # x, y = shuffle(x, y, random_state=random_state)
    # y = y.astype(int)

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=np.random.RandomState(0))
    C=1.0
    names = [
        # "svc",
        "Decision_Tree",
        "Random_Forest",
        "NeuralNetwork",
        "logistic_regression"

    ]

    classifiers = [
        # SVC(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state, class_weight='balanced'),
        RandomForestClassifier(random_state=random_state, class_weight='balanced'),
        MLPClassifier(random_state=random_state),
        linear_model.LogisticRegression( random_state=random_state, class_weight='balanced')



    ]

    from sklearn.model_selection import GridSearchCV
    for name, clf in zip(names, classifiers):

        if name=="Decision_Tree":
            df2 = df.sample(frac=1)
            y = df2[response]
            x = df2.drop(response, 1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3,
                                                                random_state=random_state)

            param = {'splitter': ['best', 'random'], 'max_features': ['auto', 'sqrt', 'log2', None], 'max_depth':[10,20,50,100, None], 'min_samples_leaf':[1, 5, 10, 50]}
            clf = GridSearchCV(clf, param, scoring=scorer)
            models.append(clf)
            names.append(name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print name," precision:", precision," recall:", recall," f1:", f1," auc:", auc, " tn:", tn," fp:", fp," fn:", fn," tp:", tp
        if name=="NeuralNetwork":
            df2 = df.sample(frac=0.25)
            y = df2[response]
            x = df2.drop(response, 1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3,
                                                                random_state=random_state)
            param = {'hidden_layer_sizes': [3,6,10,50], 'activation': ['identity', 'logistic', 'tanh', 'relu']}
            clf = GridSearchCV(clf, param, scoring=scorer)
            models.append(clf)
            names.append(name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            print name, " precision:", precision, " recall:", recall, " f1:", f1, " auc:", auc, " tn:", tn, " fp:", fp, " fn:", fn, " tp:", tp
        if name=="svc":
            df2 = df.sample(frac=0.01)
            y = df2[response]
            x = df2.drop(response, 1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3,
                                                                random_state=random_state)

            param={'kernel': ['linear', 'poly', 'rbf'], 'C': [1, 10]}
            clf = GridSearchCV(clf, param, scoring=scorer)
            models.append(clf)
            names.append(name)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc=roc_auc_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred)
            tn, fp, fn, tp=confusion_matrix(y_test, y_pred).ravel()

            print name, " precision:", precision, " recall:", recall, " f1:", f1, " auc:", auc, " tn:", tn, " fp:", fp, " fn:", fn, " tp:", tp


        if name=="Random_Forest":
            df2 = df.sample(frac=1)
            y = df2[response]
            x = df2.drop(response, 1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3,
                                                                random_state=random_state)
            param={'n_estimators': [10,50,100,200], 'max_depth': [20,30,None]}
            clf2 = GridSearchCV(clf, param, scoring=scorer)
            models.append(clf2)
            names.append(name)
            clf2.fit(X_train, y_train)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            print name, " precision:", precision, " recall:", recall, " f1:", f1, " auc:", auc, " tn:", tn, " fp:", fp, " fn:", fn, " tp:", tp
            naming = list(X_train)
            feature_df = pd.DataFrame(clf.feature_importances_, columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            feature_df.to_csv(path_or_buf='defection_model_features_rf.txt', index=True)

        elif name == "logistic_regression":
            df2 = df.sample(frac=1)
            y = df2[response]
            x = df2.drop(response, 1)
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3,
                                                                random_state=random_state)

            clf.fit(X_train, y_train)
            models.append(clf)
            names.append(name)
            y_pred = clf.predict(X_test)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            print name, " precision:", precision, " recall:", recall, " f1:", f1, " auc:", auc, " tn:", tn, " fp:", fp, " fn:", fn, " tp:", tp

            # naming = list(X_train)
            # feature_df = pd.DataFrame(clf.coef_[0], columns=['sig'], index=naming).abs().sort_values(['sig'],ascending=False)
            # feature_df2 = pd.DataFrame(clf.coef_[0], columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            # feature_df2.to_csv(path_or_buf='defection_model_features.txt', index=True)
            # feature_df2.reset_index(['naming'], inplace=True)
            #
            #
            # top_features=feature_df2[np.exp(feature_df2['sig']) >=1.1 ]
            #
            # top_df = pd.concat([x[top_features['index'].tolist()], y], axis=1)
            # top_df2 = top_df.sample(frac=0.01)
            # # print 'Features with >1.1 odds ratio', list(top_df)
            # y = top_df2[response]
            # x = top_df2.drop(response, 1)
            #
            # x, y = shuffle(x, y, random_state=random_state)
            # y = y.astype(int)
            # poly = PolynomialFeatures(3)
            # r = poly.fit_transform(x)
            #
            # feature_interaction=poly.get_feature_names(list(x))
            # df=DataFrame(r, columns=feature_interaction)
            #
            #
            #
            # X_train2, X_test2, y_train2, y_test2 = train_test_split(df, y, test_size=.3,random_state=random_state)
            # reg=linear_model.LogisticRegression()
            # reg.fit(X_train2, y_train2)
            # naming = list(X_train2)
            # feature_df2 = pd.DataFrame(reg.coef_[0], columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
            # feature_df2.to_csv(path_or_buf='defection_model_segments.txt', index=True)
            # # print feature_df2
            # y_pred2 = reg.predict(X_test2)
            # precision = average_precision_score(y_test2, y_pred2)
            # recall = recall_score(y_test2, y_pred2)
            # auc = roc_auc_score(y_test2, y_pred2)
            # print "polynomial Logistic Regression", "Precision:", precision," Recall:", recall, " ROC_AUC:", auc


        filename = 'finalized_model.sav'
        pickle.dump(models, open(filename, 'wb'))
        filename2 = 'name.sav'
        pickle.dump(names, open(filename2, 'wb'))


        # return name, ' model', ' precision score', precision, ' recall score', recall, ' f1 ', f1




    return models, names

