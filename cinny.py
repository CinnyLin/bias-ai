'''
Evaluation functions
'''

import pandas as pd
import matplotlib.pyplot as plt


# ---------- model selection ----------------
def train_test_split_pro(X, y, test_size, col='race_num'):
    '''
    train_test_split proportion to col; assumes binary col
    '''
    # get filter data
    X_col = X.columns
    y_col = y.name
    
    df = pd.merge(X, y, left_index=True, right_index=True)
    df0 = df[df[col]==0]
    df1 = df[df[col]==1]
    
    X0 = df0[X_col]
    X1 = df1[X_col]
    y0 = df0[y_col]
    y1 = df1[y_col]
    
    '''get train test data'''
    # random sample
    X_test0 = X0.sample(frac=test_size, random_state=42)
    X_test1 = X1.sample(frac=test_size, random_state=42)
    y_test0 = y0.sample(frac=test_size, random_state=42)
    y_test1 = y1.sample(frac=test_size, random_state=42)
    
    # training data = full data - testing data
    X_train0 = X0[~X0.isin(X_test0)].dropna()
    X_train1 = X1[~X1.isin(X_test0)].dropna()
    y_train0 = y0[~y0.isin(X_test0)].dropna()
    y_train1 = y1[~y1.isin(X_test0)].dropna()
    
    # combine proportions
    X_test = pd.concat([X_test0, X_test1])
    X_train = pd.concat([X_train0, X_train1])
    y_test = pd.concat([y_test0, y_test1])
    y_train = pd.concat([y_train0, y_train1])
    
    return X_train, X_test, y_train, y_test

# ----------- data preprocessing --------------

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


def plot_scatter(df, truth_label='recidivism_within_2_years', \
                 pred_label='COMPASS_determination', \
                 x_col='age', y_col='priors_count', \
                 true_positive=True, true_negative=True, \
                 false_positive=True, false_negative=True, \
                 race=False):
    
    # filter by race
    if race==True:
        # get filters
        tp, tn, fp, fn = get_filters(df, truth_label, pred_label)
    
        # get race filters
        try:
            r1 = (df['race']=='Caucasian')
            r2 = (df['race']=='African-American')
        except KeyError:
            r1 = (df['race_num']==0) #Caucasian, Asian etc
            r2 = (df['race_num']==1) #African-American, Hispanic
            
        # plot
        plt.figure(figsize=(10,5))
#         if true_positive==True:
#             plt.scatter(TP[x_col], TP[y_col], label='true positive')
#         if true_negative==True:
#             plt.scatter(TN[x_col], TN[y_col], label='true negative')
        if false_positive==True:
            plt.scatter(df[fp&r1][x_col], df[fp&r1][y_col], label='false positive -- Caucasian group')
        if false_negative==True:
            plt.scatter(df[fn&r1][x_col], df[fn&r1][y_col], label='false negative -- Caucasian group')
        if false_positive==True:
            plt.scatter(df[fp&r2][x_col], df[fp&r2][y_col], label='false positive -- African American group')
        if false_negative==True:
            plt.scatter(df[fn&r2][x_col], df[fn&r2][y_col], label='false negative -- African American group')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
    
    
    else: # race==False
        # get data
        TP, TN, FP, FN = get_data(df, truth_label, pred_label)
        
        # plot
        plt.figure(figsize=(10,5))
        if true_positive==True:
            plt.scatter(TP[x_col], TP[y_col], label='true positive')
        if true_negative==True:
            plt.scatter(TN[x_col], TN[y_col], label='true negative')
        if false_positive==True:
            plt.scatter(FP[x_col], FP[y_col], label='false positive')
        if false_negative==True:
            plt.scatter(FN[x_col], FN[y_col], label='false negative')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()