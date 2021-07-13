# bias-ai

## Project Title
**Mitigating Biases in Predicting Crime Recidivism in the U.S.**


## Team
- **Students**: Cinny Lin, Mengjie Shen
- **Advisor**: Professor Bruno Abrahao

## Abstract
As two data science Chinese students who grew up in an ethnically homogenous environment until enrolling in a U.S. college, we would like to further understand issues of racism in the U.S. using analytical skills we have acquired through related coursework. More specifically, we would like to use different algorithms and metrics from IBM, Google, and our advisor, to identify and reduce racial biases in crime recidivism prediction in different U.S. states. 

## Research Question and Significance
Black Lives Matter is a decentralized political and social movement that started in 201, but issues around racism have only exacerbated in the United States. Amidst COVID-19 pandemic last year, George Floyd's death triggered a series of ongoing protests about police brutality against black people. As two data science students who grew up in Greater China and were never really exposed to people with different ethnicities until enrolling in a U.S. college, we would like to further understand racial bias in the U.S. using an analytical approach. 

More specifically, we would like to mitigate biases in algorithms for predicting crime recidivism in the U.S. From our preliminary investigation, we learned that different states adopt their respective academic or commercial tools to predict the probability of criminals reoffending. For example, non-profit organization ProPublica has assessed the commercial tool Commercial Offender Management Profiling for Alternative Sanctions (COMPAS) NorthPointe developed for Florida, and discovered that even when controlling for prior crimes, future recidivism, age, and gender, black defendants were 77 percent more likely to be assigned higher risk scores than white defendants. 

## Project Design and Feasibility
First, we would like to use the [crime dataset from Broward County, Florida](https://www.kaggle.com/danofer/compass), as the baseline and further explore more datasets to identify whether bias in recidivism prediction is perceptible in other counties and states in the U.S. Currently, we have found datasets available on kaggle on the [recidivism statistics in New York City](https://www.kaggle.com/new-york-state/nys-recidivism-beginning-2008?select=recidivism-beginning-2008.csv) as well as the [recidivism statistics in the State of Iowa](https://www.kaggle.com/slonnadube/recidivism-for-offenders-released-from-prison) for further exploration. 

Then, we would like to use the various methods that [IBM's AI Fairness 360](http://aif360.mybluemix.net) and [Google's What-If Tools](https://pair-code.github.io/what-if-tool/) have to offer, addressing bias throughout AI systems using algorithms such as optimized pre-processing, reweighing, and disparate impact remover, and finding out whether different individuals and groups are treated fairly using metrics such as statistical parity difference, Theil index, and Euclidean distance. We would also like to use the algorithm to identify unknown unknowns that Professor Abrahao has developed in [this paper](https://stern.hosting.nyu.edu/abrahao/supervised-discovery-of-unknown-unknowns-through-test-sample-mining/). Eventually, the goal of our research is to see the effectiveness of each and aggregated algorithms in reducing prediction biases based on social demographics, such as but not limited to age, gender, and race. 