'''
Evaluation functions
'''

import pandas as pd

# ------------ get preliminary data ------------

def get_filters(df, truth_label='recidivism_within_2_years', pred_label='COMPASS_determination'):
    tp = ((df[truth_label]==1) & (df[pred_label]==1))
    tn = ((df[truth_label]==0) & (df[pred_label]==0))
    fp = ((df[truth_label]==0) & (df[pred_label]==1))
    fn = ((df[truth_label]==1) & (df[pred_label]==0))
    return tp, tn, fp, fn

def get_data(df, truth_label='recidivism_within_2_years', pred_label='COMPASS_determination'):
    tp, tn, fp, fn = get_filters(df, truth_label, pred_label)
    tp = df[tp]
    tn = df[tn]
    fp = df[fp]
    fn = df[fn]
    return tp, tn, fp, fn

def get_length(df, truth_label='recidivism_within_2_years', pred_label='COMPASS_determination'):
    tp, tn, fp, fn = get_data(df, truth_label, pred_label)
    return len(tp), len(tn), len(fp), len(fn)


# -------------- get evaluation metrics ---------------

def get_accuracy(df, truth_label='recidivism_within_2_years', pred_label='COMPASS_determination'):
    '''
    Accuracy is a valid choice of evaluation for classification problems which are well balanced and not skewed or No class imbalance.
    '''
    tp, tn, fp, fn = get_data(df, truth_label, pred_label)
    TP, TN, FP, FN = get_length(df, truth_label, pred_label)
    return (TP+TN)/(TP+FP+FN+TN)

def get_precision(df, truth_label='recidivism_within_2_years', pred_label='COMPASS_determination'):
    '''
    Precision is a valid choice of evaluation metric when we want to be very sure of our prediction.
    '''
    tp, tn, fp, fn = get_data(df, truth_label, pred_label)
    TP, TN, FP, FN = get_length(df, truth_label, pred_label)
    return (TP)/(TP+FP)

def get_recall(df, truth_label='recidivism_within_2_years', pred_label='COMPASS_determination'):
    '''
    Recall is a valid choice of evaluation metric when we want to capture as many positives as possible.
    '''
    tp, tn, fp, fn = get_data(df, truth_label, pred_label)
    TP, TN, FP, FN = get_length(df, truth_label, pred_label)
    return (TP)/(TP+FN)

def get_f1(df, truth_label='recidivism_within_2_years', pred_label='COMPASS_determination'):
    '''
    The F1 score is a number between 0 (worst) and 1 (best). It is used when you want your model to have both good precision and recall.
    '''
    P = get_precision(df, truth_label, pred_label)
    R = get_recall(df, truth_label, pred_label)
    return 2*(P*R)/(P+R)


# --------------- analysis ---------------

def propublica_analysis(df, truth_label='recidivism_within_2_years', \
                        pred_label='COMPASS_determination', \
                        race1='Caucasian', race2='African-American'):
    '''
    Duplicate Propublilca Analysis
    Prediction Fails Differently for Black Defendants
    '''
    
    # create filters
    tp, tn, fp, fn = get_filters(df, truth_label, pred_label)
    try:
        r1 = (df['race']==race1)
        r2 = (df['race']==race2)
    except KeyError:
        r1 = (df['race_num']==race1)
        r2 = (df['race_num']==race2)
    
    # get lengths
    fp1 = len(df[fp&r1])
    fn1 = len(df[fn&r1])
    fp2 = len(df[fp&r2])
    fn2 = len(df[fn&r2])
    
    # return probabilities
    try:
        p1 = fp1/(fp1+fn1)
    except ZeroDivisionError:
        p1 = 0
    
    try:
        p2 = fn1/(fp1+fn1)
    except ZeroDivisionError:
        p2 = 0
    
    try:
        p3 = fp2/(fp2+fn2)
    except ZeroDivisionError:
        p3 = 0
    
    try:
        p4 = fn2/(fp2+fn2)
    except ZeroDivisionError:
        p4 = 0
    
    return p1, p2, p3, p4