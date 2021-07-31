import streamlit as st
from st_footer import footer

import numpy as np
import pandas as pd
from models import metrics, logit


st.markdown('# Mitigating Biases in Predicting Crime Recidivism in the U.S.')
st.write('Dean\'s Undergraduate Research Project at NYU Shanghai, Summer 2021')

# st.markdown('## Team')
st.write('''
         - **Students**: Cinny Lin, Mengjie Shen
         - **Advisor**: Professor Bruno Abrahao
         ''')

st.markdown('## Project')
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
abstract_section.write(abstract)

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
significance_section.write(significance)

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
design_section.write(design)


st.markdown('---')


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
                - In `sex`, "Male" is encoded as `1`, and "Female" as `0`
                - In `race`, "African American" and "Hispanic" is encoded as `1`,
                and "Asian", "Caucasian", "Native American" and "Other" are encoded as `0`
                '''
convert_cat_section = st.beta_expander("Converting Categorical Variables", False)
convert_cat_section.markdown(convert_cat)


st.markdown('---')


st.markdown('## Models')

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
metric_interpret_section = st.beta_expander("Metrics and Interpretation", False)
metric_interpret_section.write(metric_interpret)

st.markdown('### Baseline Model: COMPAS')
baseline_accuracy = metrics.get_accuracy(df)
baseline_precision = metrics.get_precision(df)
baseline_recall = metrics.get_recall(df)
st.markdown(f'''
            - **Baseline accuracy** = {baseline_accuracy:.5f}
            - **Baseline precision** = {baseline_precision:.5f}
            - **Baseline recall** = {baseline_recall:.5f}
            ''')

st.markdown('### Machine Learning Models')
models = ['Logistic Regression', 'Decision Trees', 'Random Forest'
          'Naïve Bayes', 'Stochastic Gradient Descent',
          'Support Vector Machine', 'K Nearest Neighbors',
          'Artificial Neural Network']
model_name = st.selectbox("", options=models)

if model_name == 'Logistic regression':
        pass


st.markdown('---')
st.markdown('## Distribution of Work')
work = '''
        - **Cinny Lin**: logit, decision tree, random forest, ANN, models;
        metric class functions and baseline comparisons, streamlit app
        - **Menjie Shen**: naive bayes, SGD, SVM, kNN models
        '''
st.write(work)
# st.image('contributions.png')


footer("&copy; Cinny Lin. 2021.")