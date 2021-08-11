# Load all necessary packages
%matplotlib inline
import sys
sys.path.append("../")
import numpy as np
from tqdm import tqdm

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from IPython.display import Markdown, display
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

# from common_utils import compute_metrics
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
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
from sklearn.model_selection import GridSearchCV

def RW_logit(X,y):

def logit(X, y):
    lr = LogisticRegression()
    y_pred = cross_val_predict(lr, X, y, cv=10)
    y_pred_prob = cross_val_predict(lr, X, y, cv=10, method='predict_proba')[:,1]
    return y_pred, y_pred_prob

def GNB(X, y):
    gnb = GaussianNB()
    y_pred = cross_val_predict(gnb, X, y, cv = 10)
    return y_pred

def SGD(X, y):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    y_pred = cross_val_predict(clf, X, y, cv =10)
    return y_pred

def KNN(X, y):
    neigh = KNeighborsClassifier()
    y_pred = cross_val_predict(neigh, X, y, cv=10)
    return y_pred

def SVM(X, y):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    y_pred = cross_val_predict(clf, X, y, cv=10)
    return y_pred

def RF(X, y):
    # param_test1= {'n_estimators':range(100,200,5)}  
    # gsearch1= GridSearchCV(estimator = RandomForestClassifier(min_samples_split=10,  
    #                              min_samples_leaf=2,max_depth=10 ,bootstrap = True, max_features='sqrt' ,random_state=42),param_grid =param_test1, scoring='accuracy',cv=5)  
    clf = RandomForestClassifier(n_estimators = 144,
 min_samples_split = 10,
 min_samples_leaf = 2,
 max_features = 'sqrt',
 max_depth = 10,
 bootstrap = True,
 )
    y_pred = cross_val_predict(clf, X, y, cv=10)
    # gsearch1.fit(X,y)
    # y_pred = gsearch1.predict(X)
    return y_pred 

def DT(X, y):
    clf = tree.DecisionTreeClassifier()
    y_pred = cross_val_predict(clf, X, y, cv=10)
    return y_pred

def ANN(X, y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
    y_pred = cross_val_predict(clf, X, y, cv=10)
    return y_pred

