import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_predict


def logit(df, X, y):
    # model prediction
    lr = LogisticRegression()
    y_pred = cross_val_predict(lr, X, y, cv=10)
    
    # data prep
    df['Logistic Regression'] = y_pred
    return df
