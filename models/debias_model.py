import sys
sys.path.append("../")
from sklearn.metrics import accuracy_score
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing


#load data
import pandas as pd

def load_data(df, X, y):
    dataset_orig = StandardDataset(df,label_name='recidivism_within_2_years', favorable_classes=[1], protected_attribute_names=['race'], privileged_classes = [['Caucasian']], features_to_keep=X.columns.tolist())

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

    #Get the dataset and split into train and test
    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
#     dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    return dataset_orig_train, dataset_orig_vt

def preprocessing_Reweighing(df, X, y):

    data = StandardDataset(df,label_name='recidivism_within_2_years', favorable_classes=[1], protected_attribute_names=['race'], privileged_classes = [['Caucasian']], features_to_keep=X.columns.tolist())
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
df = pd.read_csv('data/compas_florida.csv')
X= df[['sex_num', 'age', 'African_American', 'Caucasian','priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']]
y = df['recidivism_within_2_years']
# y_pred = inprocessing_aversarial_debaising(df, X, y)
# print(print(y_pred))

def inprocessing_aversarial_debaising(df, X, y):
    data = load_data(df, X, y)
    tf.disable_eager_execution()
    dataset_orig_train = data[0]
    dataset_orig_vt = data[1]
#     print(dataset_orig_vt)
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()    
    X_vt = scale_orig.fit_transform(dataset_orig_vt.features)
    y_vt = dataset_orig_vt.labels.ravel()    
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    tf.reset_default_graph()
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,unprivileged_groups = unprivileged_groups,scope_name='debiased_classifier',debias=True,sess=sess)
    debiased_model.fit(dataset_orig_train)
    y_pred = debiased_model.predict(dataset_orig_vt).labels.ravel()
    
    # data_orig_vt = dataset_orig_vt.convert_to_dataframe()
    return y_vt, y_pred


def postprocessing_calibrated_eq_odd(df,X, y):
    data = load_data(df, X, y)
    dataset_orig_train = data[0]
    dataset_orig_vt = data[1]
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_orig_train.features)
    y_train = dataset_orig_train.labels.ravel()    
    X_vt = scale_orig.fit_transform(dataset_orig_vt.features)
    y_vt = dataset_orig_vt.labels.ravel()  
    dataset_orig_train_pred = dataset_orig_train.copy(deepcopy=True)
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    cost_constraint = "fnr" 
    randseed = 12345679 
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                        unprivileged_groups = unprivileged_groups,
                                        cost_constraint=cost_constraint,
                                        seed=randseed)
    cpp = cpp.fit(dataset_orig_train,  dataset_orig_train_pred)
    y_pred = cpp.predict(dataset_orig_vt).labels.ravel()
    print(accuracy_score(y_vt, y_pred))
    return y_vt, y_pred
y_pred = postprocessing_calibrated_eq_odd(df, X, y)
print(y_pred)




