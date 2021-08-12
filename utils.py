import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from models import metrics, models 

def plot_line_model(df, model_names, precision=False):
    '''
    Line plot for model evaluation
    '''
    
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
    
    if precision==True:
        plt.plot(model_names, model_precisions, label='precision', color='#2E4FBD')
        plt.axhline(baseline_precision, alpha=0.8, linestyle='--', color='#6495ED', label='baseline precision')
    
    ax.legend()
    return fig

def plot_line_fairness(df, model_names, 
                       dp=False, eop=False, eod=False, ca=False):
    '''
    Line plot for fairness evaluation
    '''
    
    fig, ax = plt.subplots(figsize=(20,7))
    
    # get data
    bsl_dp, bsl_eop, bsl_eod, bsl_ca = metrics.fairness_metrics(df)
    
    mdl_dps = []
    mdl_eops = []
    mdl_eods = []
    mdl_cas = []

    for model in model_names:
        mdl_dp, mdl_eop, mdl_eod, mdl_ca = metrics.fairness_metrics(df, pred_label=model)
        mdl_dps.append(mdl_dp)
        mdl_eops.append(mdl_eop)
        mdl_eods.append(mdl_eod)
        mdl_cas.append(mdl_ca)
    
    # plot
    if dp:
        plt.plot(model_names, mdl_dps, label='demographic parity', color='#2E4FBD') #blue
        plt.axhline(bsl_dp, alpha=0.8, linestyle='--', color='#6495ED', label='baseline demographic parity')
        
    if eop:
        plt.plot(model_names, mdl_eops, label='equal opportunity', color='#D64949') #red
        plt.axhline(bsl_eop, alpha=0.8, linestyle='--', color='#FF5733', label='baseline equal opportunity')
    
    if eod:
        plt.plot(model_names, mdl_eods, label='equalized odds', color='#097969') #green
        plt.axhline(bsl_eod, alpha=0.8, linestyle='--', color='#2AAA8A', label='baseline equalized odds')
    
    if ca:
        plt.plot(model_names, mdl_cas, label='calibration', color='#702963') #purple
        plt.axhline(bsl_ca, alpha=0.8, linestyle='--', color='#AA336A', label='baseline calibration')
    
    ax.legend()
    return fig

def plot_line_process(df, model_accuracies, model_precisions, precision=False):
    fig, ax = plt.subplots(figsize=(20,7))
    
    # get data
    baseline_accuracy = metrics.get_accuracy(df)
    baseline_precision = metrics.get_precision(df)
    
    model_names = ['Pre-processing', 'In-processing', 'Post-processing']
    
    plt.plot(model_names, model_accuracies, label='accuracy', color='#D64949')
    plt.axhline(baseline_accuracy, alpha=0.8, linestyle='--', color='#FF5733', label='baseline accuracy')
    
    if precision==True:
        plt.plot(model_names, model_precisions, label='precision', color='#2E4FBD')
        plt.axhline(baseline_precision, alpha=0.8, linestyle='--', color='#6495ED', label='baseline precision')
    
    ax.legend()
    return fig


def plot_scatter(df, y='Logistic Regression Prob', threshold=0.5):
    race_filter1 = df['African_American']==1
    race_filter2 = df['Caucasian']==1
    truth_filter = df['recidivism_within_2_years']==1
    prob_filter = df[y]>=threshold
    
    fig, _ = plt.subplots(nrows=2, ncols=2,  
                          sharex=True, sharey=True, 
                          figsize=(20, 20))
    plt.subplots_adjust(wspace=0.05, hspace=0.05) 

    ax1 = plt.subplot(2, 2, 1)
    df1 = df[race_filter1&truth_filter]
    ax1.plot(df1[prob_filter][y], 'o', c='#C0392B')
    ax1.plot(df1[(~prob_filter)][y], 'o', c='#2874A6')
    ax1.set_xticks([])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_title('African American')

    ax2 = plt.subplot(2, 2, 2)
    df2 = df[(race_filter2)&truth_filter]
    ax2.plot(df2[prob_filter][y], 'o', c='#C0392B')
    ax2.plot(df2[(~prob_filter)][y], 'o', c='#2874A6')
    ax2.set_xticks([])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_title('Caucasian')

    ax3 = plt.subplot(2, 2, 3)
    df3 = df[race_filter1&(~truth_filter)]
    ax3.plot(df3[prob_filter][y], 'o', c='#C0392B')
    ax3.plot(df3[(~prob_filter)][y], 'o', c='#2874A6')
    ax3.set_xticks([])

    ax4 = plt.subplot(2, 2, 4)
    df4 = df[(race_filter2) & (~truth_filter)]
    ax4.plot(df1[prob_filter][y], 'o', c='#C0392B')
    ax4.plot(df1[(~prob_filter)][y], 'o', c='#2874A6')
    ax4.set_xticks([])
    ax4.set_yticks([0, 0.5, 1])

    fig.text(0.09, 0.5, 'Re-offending Probability', va='center', rotation='vertical')
    fig.text(0.92, 0.66, 'Ground Truth Re-offended', ha='center', rotation='vertical')
    fig.text(0.92, 0.3, 'Ground Truth Did Not Re-offend', va='center', rotation='vertical')
    
    return fig


def plot_heatmap(model_names, data, vals,
                 row_labels=["False Positive", "False Negative"], col_labels=["White", "Black"]):
            
    ## get min/max
    data_min = min(vals)
    data_max = max(vals)
    
    # plot
    fig, ax = plt.subplots(nrows=2, ncols=4,  sharex=True, sharey=True, figsize=(10, 10))
    plt.subplots_adjust(top=4.5, bottom=3, right=6, left=3.5, wspace=0.05, hspace=0.05)
    
    fontsize = 20
    
    for i in range(8):
        ax = plt.subplot(2, 4, i+1)
        im = ax.imshow(data[i], cmap="YlOrRd",
                       vmin=data_min, vmax=data_max)

        annotate_heatmap(im, valfmt="{x:.0f}%", size=40)

        ax.set_xticks(np.arange(data[i].shape[1]))
        ax.set_yticks(np.arange(data[i].shape[0]))
        ax.set_xticklabels(col_labels, size=fontsize)
        ax.set_yticklabels(row_labels, size=fontsize)
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

        if i == 0:
            ax.set_xlabel('Baseline COMPAS', size=fontsize)
        else:
            ax.set_xlabel(model_names[i-1], size=fontsize)

        ax.spines[:].set_visible(False) # remove grid

    return fig


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts