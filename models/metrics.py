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
                        pred_label='COMPASS_determination'):
    '''
    Duplicate Propublilca Analysis
    Prediction Fails Differently for Black Defendants
    source: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
    '''
    
    # create label filters
    tp, tn, fp, fn = get_filters(df, truth_label, pred_label)
    
    # create race filters
    try:
        r1 = (df['race']=='Caucasian')
        r2 = (df['race']=='African-American')
    except:
        r1 = (df['Caucasian']==1)
        r2 = (df['African_American']==1)
    
    # get lengths
    fp1 = len(df[fp&r1])
    fn1 = len(df[fn&r1])
    fp2 = len(df[fp&r2])
    fn2 = len(df[fn&r2])
    len1 = len(df[r1])
    len2 = len(df[r1])
    
    # get probabilities
    try:
        # p1 = fp1/(fp1+fn1)
        p1 = fp1/len1
    except ZeroDivisionError:
        p1 = 0
    
    try:
        # p2 = fn1/(fp1+fn1)
        p2 = fn1/len1
    except ZeroDivisionError:
        p2 = 0
    
    try:
        # p3 = fp2/(fp2+fn2)
        p3 = fp2/len2
    except ZeroDivisionError:
        p3 = 0
    
    try:
        # p4 = fn2/(fp2+fn2)
        p4 = fn2/len2
    except ZeroDivisionError:
        p4 = 0
        
    # return percentages
    def get_percent(p):
        return int(round(p,2)*100)
    
    return [get_percent(p) for p in [p1, p2, p3, p4]]


def fairness_metrics(df, truth_label='recidivism_within_2_years', \
                    pred_label='COMPASS_determination'):
    '''
    Returns four fairness metrics:
    1. demographic parity
    2. equal opportunity
    3. equalized odds
    4. calibration
    '''
    # create label filters
    tp, tn, fp, fn = get_filters(df, truth_label, pred_label)
    
    # create race filters
    try:
        r1 = (df['race']=='Caucasian')
        r2 = (df['race']=='African-American')
    except:
        r1 = (df['Caucasian']==1)
        r2 = (df['African_American']==1)
    
    # get lengths
    # false
    fp1 = len(df[fp&r1])
    fn1 = len(df[fn&r1])
    fp2 = len(df[fp&r2])
    fn2 = len(df[fn&r2])
    # true
    tp1 = len(df[tp&r1])
    tn1 = len(df[tn&r1])
    tp2 = len(df[tp&r2])
    tn2 = len(df[tn&r2])
    # all
    len1 = len(df[r1])
    len2 = len(df[r1])
    
    '''
    1. demographic parity:
    proportion of positive decision should be the same across all groups.
    '''
    pos1 = (fp1+tp1)/len1
    pos2 = (fp2+tp2)/len2
    
    try:
        demographic_parity = pos1/pos2
        
        if demographic_parity>1:
            demographic_parity = pos2/pos1
    
    except ZeroDivisionError:
        demographic_parity = 0
    
    '''
    2. equal opportunity:
    "true negative rate" (TNR) should be equal for all groups
    '''
    tnr1 = tn1/len1
    tnr2 = tn2/len2
    
    try:
        equal_opportunity = tnr1/tnr2
        
        if equal_opportunity>1:
            equal_opportunity = tnr2/tnr1
    
    except ZeroDivisionError:
        equal_opportunity = 0
    
    '''
    3. equalized odds: 
    "false negative rate" (FNR) and "true negative rate" (TNR) should be equal across groups
    '''
    fnr1 = fn1/len1
    fnr2 = fn2/len2
    
    try:
        equalized_odds = fnr1/fnr2
    
        if equalized_odds > 1:
            equalized_odds = fnr2/fnr1
    
    except ZeroDivisionError:
        equalized_odds = 0
    
    '''
    4. calibration
    model's predicted probability should be "correct" across all groups
    '''
    correct_positive_rate1 = (tp1+fn1)/len1
    correct_positive_rate2 = (tp2+fn2)/len2
    predicted_positive_rate1 = (tp1+fp1)/len1
    predicted_positive_rate2 = (tp2+fp2)/len2
    
    calibration1 = predicted_positive_rate1/correct_positive_rate1
    calibration2 = predicted_positive_rate2/correct_positive_rate2
    
    try:
        calibration = calibration1/calibration2
        
        if calibration>1:
            calibration = calibration2/calibration1
    
    except ZeroDivisionError:
        calibration = 0
    
    return demographic_parity, equal_opportunity, equalized_odds, calibration