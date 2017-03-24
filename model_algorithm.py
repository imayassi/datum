import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score,  average_precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
random_state = np.random.RandomState(0)
def algorithm(x,y):

    names = [
        # "Nearest Neighbors" ,
        # "Decision_Tree",
        "Random_Forest"
        # "NeuralNetwork"
        # "AdaBoost",
        #  "Naive Bayes",
        # "Bernouli Niave Bayes",
        # "QDA",
        # "Bagging" ,
        # "ERT",
        # "GB"
    ]

    classifiers = [
        # KNeighborsClassifier(n_neighbors=20, leaf_size=1),
        # DecisionTreeClassifier(criterion='entropy'),
        RandomForestClassifier(criterion='entropy', n_estimators=200)
        # MLPClassifier(alpha=1e-5, random_state = random_state)
        # AdaBoostClassifier(n_estimators=100),
        # GaussianNB(),
        # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True),
        # QuadraticDiscriminantAnalysis()
        # BaggingClassifier(bootstrap_features=True,random_state=np.random.RandomState(0)),
        # ExtraTreesClassifier(criterion='entropy', random_state=np.random.RandomState(0)),
        # GradientBoostingClassifier(n_estimators=1000, max_depth=10000, random_state= np.random.RandomState(0))
    ]

    models=[]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.4, random_state=np.random.RandomState(0))
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precision = average_precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc=roc_auc_score(y_test, y_pred)
        f1=f1_score(y_test, y_pred)
        tn, fp, fn, tp=confusion_matrix(y_test, y_pred).ravel()
        f1 = f1_score(y_test, y_pred)
        # f1_avg=np.mean(cross_val_score(clf, X_train, y_train, scoring="f1", cv=3, n_jobs=1))
        # auc = np.mean(cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=3, n_jobs=1))
        # precision_avg = np.mean(cross_val_score(clf, X_train, y_train, scoring="precision", cv=3, n_jobs=1))
        # recall_avg = np.mean(cross_val_score(clf, X_train, y_train, scoring="recall", cv=3, n_jobs=1))
        print name , precision, recall, f1, auc, tn, fp, fn, tp
        naming = list(X_train)
        feature_df = pd.DataFrame(clf.feature_importances_, columns=['sig'], index=naming).sort_values(['sig'],ascending=False)
        print feature_df
            # , "recall_avg ", recall_avg, "precision_avg ", precision_avg

        models.append(clf)
        # return name, ' model', ' precision score', precision, ' recall score', recall, ' f1 ', f1
    return models, names

