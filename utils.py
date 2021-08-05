import pandas as pd
import matplotlib.pyplot as plt
from models import metrics, models 

def plot_line(df, model_names, precision=False):
    fig, ax = plt.subplots(figsize=(20,7))
    
    # get data
    baseline_accuracy = metrics.get_accuracy(df)
    baseline_precision = metrics.get_precision(df)
    
    model_accuracies = []
    model_precisions = []

    for model in model_names:
        model_accuracy = metrics.get_accuracy(df, pred_label=model)
        model_precision = metrics.get_precision(df, pred_label=model)
        model_accuracies.append(model_accuracy)
        model_precisions.append(model_precision)
    
    # plot
    plt.plot(model_names, model_accuracies, label='accuracy', color='#D64949')
    plt.axhline(baseline_accuracy, alpha=0.8, linestyle='--', color='#FF5733', label='baseline accuracy')
    
    if precision:
        plt.plot(model_names, model_precisions, label='precision', color='#2E4FBD')
        plt.axhline(baseline_precision, alpha=0.8, linestyle='--', color='#6495ED', label='baseline precision')
    
    ax.legend()
    return fig

def plot_scatter(df, y='Logistic Regression Prob', threshold=0.5):
    race_filter = df['race_num']==1
    truth_filter = df['recidivism_within_2_years']==1
    prob_filter = df[y]>=threshold
    
    fig, _ = plt.subplots(nrows=2, ncols=2,  
                          sharex=True, sharey=True, 
                          figsize=(20, 20))
    plt.subplots_adjust(wspace=0.05, hspace=0.05) 

    ax1 = plt.subplot(2, 2, 1)
    df1 = df[race_filter&truth_filter]
    ax1.plot(df1[prob_filter][y], 'o', c='#C0392B')
    ax1.plot(df1[(~prob_filter)][y], 'o', c='#2874A6')
    ax1.set_xticks([])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_title('African American & Hispanic')

    ax2 = plt.subplot(2, 2, 2)
    df2 = df[(~race_filter)&truth_filter]
    ax2.plot(df2[prob_filter][y], 'o', c='#C0392B')
    ax2.plot(df2[(~prob_filter)][y], 'o', c='#2874A6')
    ax2.set_xticks([])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_title('Caucasian & others')

    ax3 = plt.subplot(2, 2, 3)
    df3 = df[race_filter&(~truth_filter)]
    ax3.plot(df3[prob_filter][y], 'o', c='#C0392B')
    ax3.plot(df3[(~prob_filter)][y], 'o', c='#2874A6')
    ax3.set_xticks([])

    ax4 = plt.subplot(2, 2, 4)
    df4 = df[(~race_filter) & (~truth_filter)]
    ax4.plot(df1[prob_filter][y], 'o', c='#C0392B')
    ax4.plot(df1[(~prob_filter)][y], 'o', c='#2874A6')
    ax4.set_xticks([])
    ax4.set_yticks([0, 0.5, 1])

    fig.text(0.09, 0.5, 'Re-offending Probability', va='center', rotation='vertical')
    fig.text(0.92, 0.66, 'Ground Truth Re-offended', ha='center', rotation='vertical')
    fig.text(0.92, 0.3, 'Ground Truth Did Not Re-offend', va='center', rotation='vertical')
    
    return fig