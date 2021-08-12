# Load all necessary packages
# %matplotlib inline
# Load all necessary packages
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
from sklearn.model_selection import cross_val_predict
from IPython.display import Markdown, display
import matplotlib.pyplot as plt

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing

# from common_utils import compute_metrics
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
        import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

#load data
import pandas as pd

def load_data(df, X, y):
    dataset_orig = StandardDataset(df,label_name='recidivism_within_2_years', favorable_classes=[1], protected_attribute_names=['race'], privileged_classes = [['Caucasian']], features_to_keep=X.columns.tolist())

#     dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
#     privileged_groups = [{'race': 1}]
#     unprivileged_groups = [{'race': 0}]

    # Get the dataset and split into train and test
#     dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
#     dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    # return dataset_orig_train, dataset_orig_valid, dataset_orig_test
    return dataset_orig

def preprocessing_Reweighing(df, X, y):
    data = load_data(df, X, y)
    # dataset_orig_train = data[0]
    # dataset_orig_valid = data[1]
    # dataset_orig_test = data[2]

    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    #preprocession algorithm-reweighing

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(data)

    # Logistic regression classifier and predictions
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(data.features)
    y_train = data.labels.ravel()
    w_train = data.instance_weights.ravel()

    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,sample_weight=data.instance_weights)
    y_pred = cross_val_predict(lmod, X_train, y_train, fit_params = {'sample_weight':w_train}, cv = 10)
#     print(y_pred)
    return y_pred
# from sklearn.metrics import accuracy_score
# df = pd.read_csv('/Users/shenmengjie/Documents/GitHub/bias-ai/data/compas_florida.csv')
# X= df[['sex_num', 'age', 'African_American', 'Caucasian','priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']]
# y = df['recidivism_within_2_years']
# y_pred = preprocessing_Reweighing(df, X,y)
# print(accuracy_score(y, y_pred))

def inprocessing_aversarial_debaising(df, X, y):
    tf.reset_default_graph()
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                                unprivileged_groups = unprivileged_groups,
                                scope_name='debiased_classifier',
                                debias=True,
                                sess=sess)
    debiased_model.fit(dataset_orig_train)


def postprocessing(df,X, y):
    cost_constraint = "fnr" 
    randseed = 12345679 
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                        unprivileged_groups = unprivileged_groups,
                                        cost_constraint=cost_constraint,
                                        seed=randseed)
    cpp = cpp.fit(dataset_orig_valid, dataset_orig_valid_pred)




