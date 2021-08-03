import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

def logit(df, X, y):
    # model prediction
    lr = LogisticRegression()
    y_pred = cross_val_predict(lr, X, y, cv=10)
    
    # data prep
    df['Logistic Regression'] = y_pred
    return df

def GaussianNB(df, X, y):
    gnb = GaussianNB()
    y_pred = cross_val_predict(gnb, X, y, cv = 10)
    
    df["Naive Bayes"] = y_pred
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

