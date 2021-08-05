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