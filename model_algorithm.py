
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score,  average_precision_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
import pickle
random_state = np.random.RandomState(0)
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
def algorithm(x,y, response):
    df=pd.concat([x,y], axis=1)
    y = df[response]
    x = df.drop(response, 1)
    models = []
    x, y = shuffle(x, y, random_state=np.random.RandomState(0))
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=np.random.RandomState(0))
    C=1.0
    names = [
        "Nearest Neighbors" ,
        "Support Vector",
        "rbf_svc",
        "poly_svc",
        "lin_svc",
        "Decision_Tree",
        "Random_Forest",
        "logistic_regression",
        "NeuralNetworkLogistic",
        "NeuralNetwork"

    ]

    classifiers = [
        KNeighborsClassifier(n_neighbors=300,weights='distance', leaf_size=1),
        SVC(kernel='linear', C=C,class_weight= {1: 0.6},random_state=np.random.RandomState(0)),
        SVC(kernel='rbf', gamma=0.7, C=C, class_weight={1: 0.6},random_state=np.random.RandomState(0)),
        SVC(kernel='poly', degree=4, C=C,  class_weight= {1: 0.6},random_state=np.random.RandomState(0)),
        LinearSVC(C=C,  class_weight= {1: 0.6}),
        DecisionTreeClassifier(criterion='entropy',class_weight= {1: 0.6}),
        RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=np.random.RandomState(0)),
        linear_model.LogisticRegression( random_state=np.random.RandomState(0),class_weight= {1: 0.6}),
        MLPClassifier(alpha=1e-5,activation='logistic', random_state = random_state),
        MLPClassifier(alpha=1e-5, random_state=random_state)

    ]




    for name, clf in zip(names, classifiers):
        clf.fit(x, y)
        y_pred = clf.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc=roc_auc_score(y_test, y_pred)
        f1=f1_score(y_test, y_pred)
        tn, fp, fn, tp=confusion_matrix(y_test, y_pred).ravel()
        score = cross_val_score(clf, x, y, scoring='average_precision', cv=5)
        cross_val = np.mean(score) * 100

        print name ,cross_val , precision, recall, f1, auc, tn, fp, fn, tp


        filename = 'finalized_model.sav'
        pickle.dump(clf, open(filename, 'wb'))
        filename2 = 'name.sav'
        pickle.dump(clf, open(filename2, 'wb'))

        models.append(clf)


    return models, names

