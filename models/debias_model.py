import sys
sys.path.append("../")

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split

import tensorflow.compat.v1 as tf
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing


#load data
import pandas as pd
import streamlit as st

def load_data(df, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    df_train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    df_test = pd.merge(X_test, y_test, left_index=True, right_index=True)
    
    # dataset_orig = StandardDataset(df,label_name='recidivism_within_2_years', favorable_classes=[1], protected_attribute_names=['race'], privileged_classes = [['Caucasian']], features_to_keep=X.columns.tolist())
    dataset_orig_train = StandardDataset(df_train,label_name='recidivism_within_2_years', favorable_classes=[1], protected_attribute_names=['Caucasian'], privileged_classes = [[1]], features_to_keep=X.columns.tolist())
    dataset_orig_vt = StandardDataset(df_test,label_name='recidivism_within_2_years', favorable_classes=[1], protected_attribute_names=['Caucasian'], privileged_classes = [[1]], features_to_keep=X.columns.tolist())

    # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    # privileged_groups = [{'race': 1}]
    # unprivileged_groups = [{'race': 0}]

    # get the dataset and split into train and test
    # dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
    # dataframe_orig_vt = dataset_orig_vt.convert_to_dataframe()
    
    return dataset_orig_train, dataset_orig_vt, df_test

def preprocessing_Reweighing(df, X, y):

    data = StandardDataset(df,label_name='recidivism_within_2_years', favorable_classes=[1], protected_attribute_names=['race'], privileged_classes = [['Caucasian']], features_to_keep=X.columns.tolist())

    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(data)

    # logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(data.features)
    y_train = data.labels.ravel()
    w_train = data.instance_weights.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,sample_weight=data.instance_weights)
    y_pred = cross_val_predict(lmod, X_train, y_train, fit_params = {'sample_weight':w_train}, cv = 10)
    return y_pred


def inprocessing_aversarial_debaising(df, X, y):
    dataset_orig_train, dataset_orig_vt, dataframe_orig_vt = load_data(df, X, y)
    tf.disable_eager_execution()
    
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()    
    X_vt = scale_orig.fit_transform(dataset_orig_vt.features)
    y_vt = dataset_orig_vt.labels.ravel()    
    privileged_groups = [{'Caucasian': 1}]
    unprivileged_groups = [{'Caucasian': 0}] #{'African_American': 1}
    tf.reset_default_graph()
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups=privileged_groups,unprivileged_groups=unprivileged_groups,scope_name='debiased_classifier',debias=True,sess=sess)
    debiased_model.fit(dataset_orig_train)
    y_pred = debiased_model.predict(dataset_orig_vt).labels.ravel()
    
    return y_vt, y_pred, dataframe_orig_vt


def postprocessing_calibrated_eq_odd(df,X, y):
    dataset_orig_train, dataset_orig_vt, dataframe_orig_vt = load_data(df, X, y)


    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()    
    X_vt = scale_orig.fit_transform(dataset_orig_vt.features)
    y_vt = dataset_orig_vt.labels.ravel()  
    privileged_groups = [{'Caucasian': 1}]
    unprivileged_groups = [{'Caucasian': 0}] #{'African_American': 1}
    cost_constraint = "fnr" 
    randseed = 12345679 


    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True) 
    dataset_orig_vt_pred = dataset_orig_vt.copy(deepcopy=True)

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_train_pred = lmod.predict(X_train)

    fav_idx = np.where(lmod.classes_ == dataset_orig_train.favorable_label)[0][0]
    y_train_pred_prob = lmod.predict_proba(X_train)[:,fav_idx]


    X_vt = scale_orig.transform(dataset_orig_vt.features)
    y_vt_pred_prob = lmod.predict_proba(X_vt)[:,fav_idx]  
    dataset_orig_vt_pred.scores = y_vt_pred_prob.reshape(-1,1)  

    # dataset_orig_train_pred.labels = y_train_pred
    dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1,1)

    class_thresh = 0.5
    y_train_pred = np.zeros_like(dataset_orig_train_pred.labels)
    y_train_pred[y_train_pred_prob >= class_thresh] = dataset_orig_train_pred.favorable_label
    y_train_pred[~(y_train_pred_prob >= class_thresh)] = dataset_orig_train_pred.unfavorable_label
    dataset_orig_train_pred.labels = y_train_pred


    y_vt_pred = np.zeros_like(dataset_orig_vt_pred.labels)
    y_vt_pred[y_vt_pred_prob >= class_thresh] = dataset_orig_vt_pred.favorable_label
    y_vt_pred[~(y_vt_pred_prob >= class_thresh)] = dataset_orig_vt_pred.unfavorable_label
    dataset_orig_vt_pred.labels = y_vt_pred

    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                        unprivileged_groups = unprivileged_groups,
                                        cost_constraint=cost_constraint,
                                        seed=randseed)
    cpp = cpp.fit(dataset_orig_train, dataset_orig_train_pred)
    
    y_pred = cpp.predict(dataset_orig_vt_pred)
    print(y_pred.labels.ravel())
    print(accuracy_score(y_pred.labels.ravel(), y_vt))

    return y_vt, y_pred.labels.ravel(), dataframe_orig_vt






