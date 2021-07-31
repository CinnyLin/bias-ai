import pandas as pd
from sklearn.linear_model import LogisticRegression 


def logit(X_train, y_train, X_test, y_test):
    # model prediction
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # data prep
    X_test['logit_v1'] = y_pred
    test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
    return test_df
