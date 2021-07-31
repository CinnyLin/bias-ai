import streamlit as st
from st_footer import footer

import numpy as np
import pandas as pd

st.markdown('# Mitigating Biases in Predicting Crime Recidivism in the U.S.')
st.write('Dean\'s Undergraduate Research Project at NYU Shanghai, Summer 2021')

# st.markdown('## Team')
st.write('''
         - **Students**: Cinny Lin, Mengjie Shen
         - **Advisor**: Professor Bruno Abrahao
         ''')

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

# st.markdown('## Project Design and Feasibility')
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
design_section = st.beta_expander("Project Design and Feasibility", False)
design_section.write(design)

# st.markdown('### Metrics and Interpretation')
metrics = '''
        The precision/recall metrics are especially interesting for us because of how it can be easily interpreted for our project.
        In a **classification** problem, **precision** is calculated by looking at the proportion 
        of true positive predictions in *all positive predictions* (true positives + false positives).
        The precision metric is a valid choice of evaluation when we want to be very sure of our prediction.
        While **recall** is calculated by looking at the propotion of true positive predictions 
        in *all actual positives* (true positives + false negatives). 
        The recall metric is a valid choice of evaluation when we want to capture as many positives as possible.
        '''
metrics_section = st.beta_expander("Metrics and Interpretation", False)
metrics_section.write(metrics)

st.markdown('---')
st.markdown('## Distribution of Work')
st.write('''
         - **Cinny Lin**: logit, decision tree, random forest, ANN, models;
             metric class functions and baseline comparisons, streamlit app
         - **Menjie Shen**: naive bayes, SGD, SVM, kNN models
         ''')
# st.image('contributions.png')


footer("&copy; Cinny Lin. 2021.")