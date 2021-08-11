# modules for app
import streamlit as st
from st_footer import footer

# modules for data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# custom modules
from models import metrics, models
import utils

title = 'Bias AI' #'Bias in Recidivism Prediction'
favicon = 'https://cdn.iconscout.com/icon/premium/png-512-thumb/bias-2979240-2483226.png'
st.set_page_config(page_title=title, page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')

st.markdown('# Analyzing Biases in Predicting Crime Recidivism in Broward, Florida')
st.write('Dean\'s Undergraduate Research Project at NYU Shanghai, Summer 2021')

# st.markdown('## Team')
st.write('''
         - **Students**: Cinny Lin, Mengjie Shen
         - **Advisor**: Professor Bruno Abrahao
         ''')

# st.markdown('## Project')
# st.markdown('## Abstract')
abstract = '''
        Cinny and Menjie are two data science students who grew up living in Taipei and Shanghai respectively,
        both of which are relatively ethnically homogenous cities, until enrolling in a U.S. univeristity.
        We would therefore like to take this summer break to further understand issues of racism in the U.S.
        using analytical skills we have acquired through related coursework at NYU.
        More specifically, we would like to use a variety of machine learning algorithms
        to identify and detect racial biases in crime recidivism prediction in different U.S. states.
        '''
abstract_section = st.beta_expander("Abstract", False)
abstract_section.markdown(abstract)

# st.markdown('## Research Question and Significance')
significance = '''
        Black Lives Matter is a decentralized political and social movement that started in 2011,
        but issues around racism have only exacerbated in the United States.
        Amidst COVID-19 pandemic in 2020, George Floyd's death triggered
        a series of ongoing protests about police brutality against black people.

        From our preliminary investigation, we learned that different states adopt their
        respective academic or commercial tools to predict the probability of criminals reoffending.
        For example, non-profit organization ProPublica has assessed the commercial tool
        Commercial Offender Management Profiling for Alternative Sanctions (COMPAS) NorthPointe developed for Florida,
        and discovered that even when controlling for prior crimes, future recidivism, age, and gender,
        black defendants were 77 percent more likely to be assigned higher risk scores than white defendants.
        '''
significance_section = st.beta_expander("Research Question and Significance", False)
significance_section.markdown(significance)

# st.markdown('## Research Design and Feasibility')
design = '''
        Due to time constraint, we have narrowed down our project scope to look at a single US region,
        and focus more on exploring the prediction accuracies of different machine learning models,
        as well as analyzing whether the prediciton results produced implicit racial biases.
        
        We have decided to use the crime dataset from Broward County, Florida, 
        and use the COMPAS commericial tool that the state has adopted as the baseline model.
        
        First, we would compare how other machine learning model **"predicts better"** (using accuracy, precision, recall metrics)
        and **"creates less biases"** (based on the table percentage from Propublica's article) than COMPAS.
        
        Second, we would look closer at the errors, look at where COMPAS predicts wrong and our model doesn't, 
        or when both do, and see the characteristics of these people.
        This would be an iterative process on top of the first step. 
        Once we identified 1-2 best models out of the 5-10 we plan to try out, 
        we can try feeding in different input variables and see how that improves the prediction.
        
        Eventually, the goal of our research project is to look to the future and give suggestions to policy changes.
        With the model and input variables we found, we are able to say, for example, 
        that race is actually not a variable that is necessary in predicting recidivism better. 
        We could say that COMPAS is not particularly useful and this is another model we propose.
        '''
design_section = st.beta_expander("Research Design and Feasibility", False)
design_section.markdown(design)


# st.markdown('---')


st.markdown('## Data')
data = '''
        We have decided to use the 
        [crime dataset](https://www.kaggle.com/danofer/compass) 
        from Broward County, Florida, 
        and use the COMPAS commericial tool that the state has adopted as the baseline model.
        '''
st.markdown(data)

df = pd.read_csv('data/compas_florida.csv')
st.dataframe(df.head(1))
st.write('Data Size:', df.shape)


convert_cat = '''
                Because `sex` and `race` are categorical variables,
                when most of our models can only take in numerical values,
                so we have one-hot-encoded them to binary variables: 
                - In `sex_num`, "Male" is encoded as `1`, and "Female" as `0`
                - Race has multiple categories, and we only one-hot-encoded the two races 
                which are the focus of our project, `African_American` and `Caucasian`
                '''
convert_cat_section = st.beta_expander("Converting Categorical Variables", False)
convert_cat_section.markdown(convert_cat)


st.markdown('### Cross Validation')
cv = '''
        We use the `cross_val_predict` method provided by 
        [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict)
        and uses 10-fold cross validation.'''
st.markdown(cv)


st.markdown('### Input Variables')

# get input columns
num_cols = df.dtypes[df.dtypes!=object].keys()
y_index = list(num_cols).index('recidivism_within_2_years')
y_col = st.selectbox(label='Target variable (y)', options=['recidivism_within_2_years'], 
                     help='The predicting variable is the ground truth value of whether the person actually reoffended 2 years later.')

#['sex_num', 'age', 'African_American', 'Caucasian',
# 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', ']
default_input_features = list(num_cols.drop([y_col, 'id', 'COMPASS_determination']))
X_cols = st.multiselect(label='Training variables (X)', options=default_input_features,
                        default=default_input_features,
                        help='Only numerical columns are visible as options.')

# get data
y = df[y_col]
X = df[X_cols]

# plot correlation matrix
XY = df[X_cols+[y_col]]
corr_matrix_section = st.beta_expander("Columns, Correlation Matrix Plot")
axis = corr_matrix_section.radio("Comparison axis:", ["column-wise", "row-wise"])
if axis=="column-wise":
        corr_fig = XY.corr().style.background_gradient(cmap='coolwarm', axis=1)
if axis=="row-wise":
        corr_fig = XY.corr().style.background_gradient(cmap='coolwarm', axis=0)
corr_matrix_section.dataframe(corr_fig)


st.markdown('## Models')

# utils
@st.cache(allow_output_mutation=True)
def run_models(model_options, df, X, y):
        for model_name in model_options:
                if model_name == 'Logistic Regression':
                        y_pred, y_pred_prob = models.logit(X, y)
                        df[model_name] =  y_pred
                        df[f'{model_name} Prob'] =  y_pred_prob
                
                elif model_name == 'Naïve Bayes':
                        y_pred = models.GNB(X, y)
                        df[model_name] =  y_pred
                
                elif model_name == "Stochastic Gradient Descent":
                        y_pred = models.SGD(X, y)
                        df[model_name] =  y_pred
                
                elif model_name == "K Nearest Neighbors":
                        y_pred = models.KNN(X, y)
                        df[model_name] =  y_pred
                
                elif model_name == "Support Vector Machine":
                        y_pred = models.SVM(X, y)
                        df[model_name] =  y_pred
                
                elif model_name == "Random Forest":
                        y_pred = models.RF(X, y)
                        df[model_name] =  y_pred
                
                elif model_name == 'Decision Trees':
                        y_pred = models.DT(X, y)
                        df[model_name] =  y_pred
                
                elif model_name == 'Artificial Neural Network':
                        y_pred = models.ANN(X, y)
                        df[model_name] =  y_pred
                
        return df

# run all models first after setting X, y 
model_options = ['Logistic Regression', 'Decision Trees', 'Random Forest',
                 'Naïve Bayes', 'Stochastic Gradient Descent',
                 'Support Vector Machine', 'K Nearest Neighbors']
                #  'Artificial Neural Network']
# df = run_models(model_options, df, X, y)

# naively dropping race variable
if ('African_American' not in X_cols) or ('Caucasian' not in X_cols):
        # df.to_csv('data/pred_output_drop_race.csv', index=False)
        df = pd.read_csv('data/pred_output_drop_race.csv')
else:
        # all input features
        # df.to_csv('data/pred_output.csv', index=False)
        df = pd.read_csv('data/pred_output.csv')
df_drop_race = pd.read_csv('data/pred_output_drop_race.csv')


# st.markdown('### Metrics and Interpretation')
metric_interpret = '''
        The precision/recall metrics are especially interesting for us because of how it can be easily interpreted for our project.
        
        In a **classification** problem, **precision** is calculated by looking at the proportion 
        of true positive predictions in *all positive predictions* (true positives + false positives).
        The precision metric is a valid choice of evaluation when we want to be very sure of our prediction.
        While **recall** is calculated by looking at the propotion of true positive predictions 
        in *all actual positives* (true positives + false negatives). 
        The recall metric is a valid choice of evaluation when we want to capture as many positives as possible.
        
        The classification problem of our project is whether a person would reoffend or not.
        That is to say, if the model has a high precision score, that means the model
        would only classify someone as "would reoffend" when the model is very certain.
        The interpretation for this is that even though people make mistakes, 
        and everyone deserves a second chance, 
        thus the model would not easily keep someone in prison.
        
        On the contrary, if the model has a high recall score, that means the model 
        would classify someone as "would reoffend" when the model thinks 
        that there is the possibility for the person to reoffend.
        The interpretation for this is that to ensure the safety of our society,
        the model would not easily release someone from prison.
        '''
metric_interpret_section = st.beta_expander("Model Evaluation: Precision and Recall Metrics", False)
metric_interpret_section.markdown(metric_interpret)

bias_interpret1 = '''
        There is currently no universal metric to detect or assess biases in a model. 
        Thus, we decided to follow the analysis that ProPublica did in its 
        [machine bias assessment article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing).
        
        One of the tables in the article calculated the percentage of 
        false positive and false negative predictions for Caucasians and African Americans
        and saw that "predictions fails differently for black defendants".
        '''
bias_interpret2 = '''
        Analyzing the same data from Broward County, Florida and the same COMPAS assessment tool,
        ProPublica found that COMPAS "correctly predicts recidivism 61 percent of the time. 
        But blacks are almost twice as likely as whites to be labeled a higher risk but not actually re-offend. 
        It makes the opposite mistake among whites: They are much more likely than blacks to be 
        labeled lower risk but go on to commit other crimes."
        
        In our project, we would use similar methods of analysis 
        and assess bias in our models by interpreting tables like the one shown above.
        '''
bias_interpret_section = st.beta_expander("Bias Evaluation: Following Propublica's Assessment", False)
bias_interpret_section.markdown(bias_interpret1)
bias_interpret_section.image('bias_table.png')
bias_interpret_section.write(bias_interpret2)

bias_discussion = '''
        There is currently no universal metric to detect or assess biases in a model. 
        
        ProPublica has found that the COMPAS tool was biased for being twice as likely to falsely predict
        African Americans as high risk when in fact they did not re-offend two years later.
        
        However, Northpointe, the company that developed the COMPAS tool, said that for any given score
        on COMPAS' 10-point scale, white and black defendants were just as likely to be predicted as would re-offend.
        
        Both claims are correct in their own terms, because they have adopted different bias assessment metrics.
        ProPublica calculated the "false positive rate", while Northpointe adopted the "calibration" method.
        
        It is also worth noting that the data that NorthPointe trained on, 
        which our project would also be training with, may embed biases.
        The likelihood of recidivism in the training data was possibly the outcome of a biased justice system.
        That would be considered label bias.
        This project focuses on model biases, assuming that we are discussing data with correct labels.
        '''
bias_discussion_section = st.beta_expander("Bias Evaluation: Discussion on Model Bias and Data Bias", False)
bias_discussion_section.markdown(bias_discussion)

bias_metrics = '''
        Notions of fairness varies and are oftentimes conflicting. 
        The following are a few ways we used in our project to compare how models performed:
        
        1. **Demographic Parity**: proportion of positive decision should be the same across all groups.
        
        _For our project, this metric means that the rate of 
        labeling a defendant as high risk should be equal across
        black and white defendants. However, this metric could cause 
        problems when the true underlying distribution 
        of risk labels differ across groups. Attempts to adjust for 
        these differences often require misclassifying
        low-risk members of one group as high-risk and vice versa, 
        potentially harming members of all groups in the process._
        
        
        2. **Equal Opportunity**: "true negative rate" (TNR) or "true positive rate" (TPR)
        should be equal for all groups, focusing on the advantaged outcome.
        
        _Our project is a negative classification problem, 
        thus focusing on looking into true negative rates. 
        For our project, this metric means that the model 
        would correctly classify defendants who do not re-offend
        as low risk at equal rates for both black and white defendants._
        
        3. **Equalized Odds**: "false negative rate" (FNR) and "true negative rate" (TNR) 
        should be equal across groups, focusing on the advantaged outcome.
        
        _Our project is a negative classification problem, 
        thus focusing on looking into true and false negative rates.
        Note that equalized odds is usually only possible by 
        introducing randomness in the decision-making procedure._
        
        4. **Calibration**: model's predicted probability should be correct across all groups.
         
        _For our project, this metric means that among 
        the defendants of a given risk score, 
        the proportion that would re-offend is the same for both groups._        
        '''
bias_metrics_section = st.beta_expander("Bias Evaluation: Fairness Metrics", False)
bias_metrics_section.markdown(bias_metrics)


st.markdown('### Baseline Model: COMPAS')

# st.markdown('#### Model Evaluation')
baseline_accuracy = metrics.get_accuracy(df)
baseline_precision = metrics.get_precision(df)
baseline_recall = metrics.get_recall(df)
baseline_eval = f'''
            - **Baseline accuracy** = {baseline_accuracy:.5f}
            - **Baseline precision** = {baseline_precision:.5f}
            - **Baseline recall** = {baseline_recall:.5f}
            '''
baseline_eval_section = st.beta_expander("Baseline Model, Model Evaluation", False)
baseline_eval_section.markdown(baseline_eval)

# st.markdown('#### Bias Evaluation')
baseline_p = metrics.propublica_analysis(df)
baseline_bias = f'''
        |COMPASS                              | White              | Black             |
        |-------------------------------------|--------------------|-------------------|
        | Labeled High Risk, Didn’t Re-Offend | {baseline_p[0]}%   | {baseline_p[2]}%  |
        | Labeled Low Risk, Yet Did Re-Offend | {baseline_p[1]}%   | {baseline_p[3]}%  |
'''
baseline_bias_section = st.beta_expander("Baseline Model, Bias Evaluation Table", False)
baseline_bias_section.markdown(baseline_bias)

# write interpretation
# st.markdown('#### Interpretation')
baseline_interpretation = f'''
        We found that COMPAS correctly predicts recidivism 
        {int(round(baseline_accuracy,2)*100)} percent of the time. 
        But blacks are almost {round(baseline_p[2]/baseline_p[0],1)} times
        as likely as whites to be labeled a higher risk but not actually re-offend. 
        It makes the opposite mistake among whites: 
        They are {round(baseline_p[1]/baseline_p[3],1)} times 
        more likely than blacks to be labeled lower risk but go on to commit other crimes.
'''
# st.markdown(baseline_interpretation)
baseline_interpretation_section = st.beta_expander("Baseline Model, Interpretation", False)
baseline_interpretation_section.markdown(baseline_interpretation)


st.markdown('### Machine Learning Models')
model_name = st.selectbox("Choose a model to assess", options=model_options)

# st.markdown('#### Model Evaluation')
model_accuracy = metrics.get_accuracy(df, pred_label=model_name)
model_precision = metrics.get_precision(df, pred_label=model_name)
model_recall = metrics.get_recall(df, pred_label=model_name)
model_eval = f'''
            - **{model_name} accuracy** = {model_accuracy:.5f}
            - **{model_name} precision** = {model_precision:.5f}
            - **{model_name} recall** = {model_recall:.5f}
            '''
model_eval_section = st.beta_expander(f"{model_name} Model, Model Evaluation", False)
model_eval_section.markdown(model_eval)

# st.markdown('#### Bias Evaluation')
model_p = metrics.propublica_analysis(df, pred_label=model_name)
model_bias = f'''
        |{model_name}                         | White              | Black             |
        |-------------------------------------|--------------------|-------------------|
        | Labeled High Risk, Didn’t Re-Offend | {model_p[0]}%      | {model_p[2]}%     |
        | Labeled Low Risk, Yet Did Re-Offend | {model_p[1]}%      | {model_p[3]}%     |
'''
model_bias_section = st.beta_expander(f"{model_name} Model, Bias Evaluation Table", False)
model_bias_section.markdown(model_bias)

# write interpretation
# st.markdown('#### Interpretation')

# (1) accuracy
model_interpretation1 = f'''
        We found that our {model_name} model correctly predicts recidivism 
        {int(round(model_accuracy,2)*100)}% of the time. '''
if model_accuracy > baseline_accuracy:
        model_interpretation11 = f'''
                *(This is a {round((model_accuracy-baseline_accuracy)/model_accuracy,3)*100}% 
                higher accuracy than the baseline model.)*'''
if model_accuracy < baseline_accuracy:
        model_interpretation11 = f'''
                *(Our model predicted worse than the baseline model.)*'''

# (2) precision & recall
if model_precision > baseline_precision:
        model_interpretation2 = f'''
                We also see that our {model_name} model has higher precision and lower recall, 
                meaning that our model was more conservative in our prediction, 
                and only predicted someone as likely to reoffend when we are very sure; 
                while the COMPAS model would predict someone as potential risk
                as long as there is the possibility that the person could reoffend.
                '''
if model_precision < baseline_precision:
        model_interpretation2 = f'''
                We also see that our {model_name} model has lower precision and higher recall, 
                meaning that our model would predict someone as potential risk
                as long as there is the possibility that the person could reoffend;
                while the COMPAS model was more conservative in its prediction, 
                and only predicted someone as likely to reoffend when it was very sure.
                '''

# (3) bias: false positive
model_interpretation3 = f'''
        But blacks are almost {round(model_p[2]/model_p[0],1)} times
        as likely as whites to be labeled a higher risk but not actually re-offend.'''

# (4) bias: false negative
model_interpretation4 = f'''
        It makes the opposite mistake among whites: 
        They are {round(model_p[1]/model_p[3],1)} times 
        more likely than blacks to be labeled lower risk but go on to commit other crimes.'''

model_interpretation_section = st.beta_expander(f"{model_name} Model, Interpretation", False)
model_interpretation_section.markdown(model_interpretation1+model_interpretation11)
model_interpretation_section.markdown(model_interpretation3+model_interpretation4)
model_interpretation_section.markdown(model_interpretation2)

# (5) bias: scatter plot
if model_name == 'Logistic Regression':
        model_scatter_section = st.beta_expander(f"{model_name} Model, Scatter Plot", False)
        threshold_slider = model_scatter_section.slider("Risk Probability Threshold:", 
                                                 min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        scatter_fig = utils.plot_scatter(df, threshold=threshold_slider)
        model_scatter_section.pyplot(scatter_fig)


# st.markdown('---')

st.markdown('## Compare Model Results')

# model_names = st.multiselect("Choose models to compare", options=model_options, default=model_options)

model_line_section = st.beta_expander("All Models, Model Evaluation Metrics, Line Plot", False)
plot_precision = model_line_section.checkbox('Also compare precision results?', value=False)
model_line_fig = utils.plot_line_model(df, model_options, precision=plot_precision)
model_line_section.pyplot(model_line_fig)

fairness_line_section = st.beta_expander("All Models, Bias Evaluation Metrics, Line Plot", False)
plot_dp = fairness_line_section.checkbox('Compare demographic parity?', value=False)
plot_eop = fairness_line_section.checkbox('Compare equal opportunity?', value=True)
plot_eod = fairness_line_section.checkbox('Compare equalized odds?', value=False)
plot_ca = fairness_line_section.checkbox('Compare calibration?', value=False)
fairness_line_fig = utils.plot_line_fairness(df, model_options,
                                             dp=plot_dp, eop=plot_eop, eod=plot_eod, ca=plot_ca)
fairness_line_section.pyplot(fairness_line_fig)

model_heatmap_section = st.beta_expander("All Selected Models, Bias Evaluation Table, Heatmap Plot", False)

# initialize with baseline compas
models_p = list()
model_p = metrics.propublica_analysis(df)
model_p_arr = np.array([[model_p[0], model_p[2]], [model_p[1], model_p[3]]])
models_p.append(model_p_arr)

vals = list()
vals.extend(model_p)

# get model probs
for model_name in model_options:
        model_p = metrics.propublica_analysis(df, pred_label=model_name)
        model_p_arr = np.array([[model_p[0], model_p[2]], [model_p[1], model_p[3]]])
        models_p.append(model_p_arr)
        vals.extend(model_p)

heatmap_fig = utils.plot_heatmap(model_options, models_p, vals)
model_heatmap_section.pyplot(heatmap_fig)

st.markdown('## Reducing Bias')

naive_reduce_bias_section = st.beta_expander('Dropping the "race" variable', False)

naive_reduce_bias_text = '''
        First, we tried the naive method of simply dropping the race_num variables,
        namely `African_American` and `Caucasian` columns from the models input features.
        The result is shown in the heatmap below, which calculates the difference of the bias tables 
        between keeping and dropping the race variables as an input feature.
        
        A possible reason why simply dropping the race variable did not reduce racial bias in the model is because
        the other input features in the model are also correlated with the race variable. In other words, even though
        we dropped the race variable itself, the other input variables that are correlated with the race variable can
        still contribute to creating racial bias for the prediction results.
        
        For example, `age` is a variable that highly negatively correlates with `African_American` 
        (as seen in the correlation matrix plot shown in the "Input Varaibles" section). 
        
        One might think, why not remove all the variables related to race then? One simple answer to this proposal
        is that many variables are correlated with the race variable.
        Also, given the constraint of the data, there are not many input features to start with,
        and dropping more features could really limit the training.
        '''
naive_reduce_bias_section.markdown(naive_reduce_bias_text)

models_p_drop_race = list()
model_p_drop_race = metrics.propublica_analysis(df_drop_race)
model_p_drop_race_arr = np.array([[model_p_drop_race[0], model_p_drop_race[2]], [model_p_drop_race[1], model_p_drop_race[3]]])
models_p_drop_race.append(model_p_drop_race_arr)

vals_drop_race = list()
vals_drop_race.extend(model_p_drop_race)

# get model probs
for model_name in model_options:
        model_p_drop_race = metrics.propublica_analysis(df_drop_race, pred_label=model_name)
        model_p_drop_race_arr = np.array([[model_p_drop_race[0], model_p_drop_race[2]], [model_p_drop_race[1], model_p_drop_race[3]]])
        models_p_drop_race.append(model_p_drop_race_arr)
        vals_drop_race.extend(model_p_drop_race)

models_p_diff = np.subtract(models_p, models_p_drop_race)
vals_diff = np.subtract(vals, vals_drop_race)

heatmap_fig_drop_race = utils.plot_heatmap(model_options, models_p_diff, vals_diff)
naive_reduce_bias_section.pyplot(heatmap_fig_drop_race)


st.markdown('## References')
references = '''
        1. Corbett-Davies & Goel's "The Measure and Mismeasure of Fairness" 
        
        2. Hardt, Price, and Srebro's "Equality of Opportunity in Supervised Learning"
        '''
st.markdown(references)

# st.markdown('---')
# st.markdown('## Distribution of Work')
# work = '''
# - **Cinny Lin**: 
#     - **Models**: logit, decision tree, random forest, ANN, models
#     - **Metrics and Interpretation**: metric class functions and baseline comparisons
#     - **Final Deliverable**: streamlit app
# \
# - **Menjie Shen**: 
#     - **Models**: naive bayes, SGD, SVM, kNN models
#         '''
# st.markdown(work)
# st.image('contributions.png')


footer("&copy; Cinny Lin. 2021.")