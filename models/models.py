import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier

def logit(df, X, y):
    # model prediction
    lr = LogisticRegression()
    y_pred = cross_val_predict(lr, X, y, cv=10)
    
    # data prep
    df['Logistic Regression'] = y_pred
    return df

def GNB(df, X, y):
    gnb = GaussianNB()
    y_pred = cross_val_predict(gnb, X, y, cv = 10)
    
    df["Naïve Bayes"] = y_pred
    return df

def SGD(df, X, y):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    y_pred = cross_val_predict(clf, X, y, cv =10)

    df["Stochastic Gradient Descent"] = y_pred
    return df

def KNN(df, X, y):
    neigh = KNeighborsClassifier()
    y_pred = cross_val_predict(neigh, X, y, cv=10)

    df["K Nearest Neighbors"] = y_pred
    return df

def SVM(df, X, y):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    y_pred = cross_val_predict(clf, X, y, cv=10)

    df['Support Vector Machine'] = y_pred
    return df

def RF(df, X, y):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    y_pred = cross_val_predict(clf, X, y, cv=10)

    df['Random Forest'] = y_pred
    return df 

def DT(df, X, y):
    clf = tree.DecisionTreeClassifier()
    y_pred = cross_val_predict(clf, X, y, cv=10)

    df['Decision Trees'] = y_pred
    return df

def ANN(df, X, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
    y_pred = cross_val_predict(clf, X, y, cv=10)

    df['Artificial Neural Network'] = y_pred
    return df

