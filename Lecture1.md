# General Introduction - 13/6/22 

* Mostly care about how well the model works for *new inputs* drawn from the same distribution. This is easy if we have infinite data, but what if we have a limited sample? 

* When we have finite data figuring out the right distribution is more difficult. This is where the Bayesian aspect comes in. We can learn a distrubtion over the parameter values conditional on whatever we have observed. 

* Predictions are made by integrating over the uncertainty.

* Still have uncertainty even as the datasets get larger. 

* Rare to be able to analytically calculate the posterior -  so we have to approximate it! :) 

How do we approximate the posterior?
* MCMC - common method but don't cover much of this in this summer school 
* Distributional approximation - slightly older method
* Flexible approximation (i.e. Normalising flow) - what a lot of the summer school will focus on 

# Introduction to Probabilistic Models - 13/6/22  
## Antonio Salmeron 
[Lecturer's notes](https://github.com/probabilisticai/probai-2022/blob/main/day_1/1_antonio/inference-probai.pdf)

Examples of probabilistic models:
* Predictive medicine 
* Self driving cars
* Image generation [imagen](https://imagen.research.google/)
* Land use - using satelite images to monitor protected areas 

These examples:
* Operate in environemnets where there's lots of data 
* Data doesn't cover every possible scenario -> uncertainty 
* Use probabilistic models 
* use inference algorithms to carry out prediction and structure analysis. 

Probabilistic models offer: 
* Principled quantification of uncertainty
* Natural way of dealing with missing data 

What is Machine Learning? [Book by Tom Mitchell, 1997](https://www.cin.ufpe.br/~cavmj/Machine%20-%20Learning%20-%20Tom%20Mitchell.pdf)

### Easy example: Linear regression
Q: Is linear regression really ML? 
[jupyter notebook](https://github.com/probabilisticai/probai-2022/blob/main/day_1/1_antonio/probAI2022.ipynb)

A: Yes, the more data we give it, the better the model - it is learning from experience.

The most basic example of linear regression is not probabilistic - but we can look at it from a Bayesian perspective. 

There are two types of uncertainty:
* **Aleatoric** - due to randomness, i.e. the variability in the outcome of an experiment due to random effects. 
* **Epistemic** - due to lack of knowledge. 

Assume we want to predict Y from X. We assume a joint distribution p(x,y) - this is Epistemic uncertainty it is unreducible. if we know p(y|x). What it p(y) for a specific x? This is aleatoric uncertainty - it is reducible. 

~Epistemic~ (?) Aleatoric uncertainty can be reduced by gathering more data, or increasing the number of features. 

Probabilitic graphical models off:
* *Structured* specification of high dimensional distributions in terms of low dimensional factors. 
* *Efficient* inference and learning taking advantage of the structure.
* *Graphical* representation is interpretable by humans. 

In a frequentist approach, would assume that the parameter is a fixed value - it is NOT a random variable. 

In a Bayesian approach, all parameters are random variables. Information about parameters can be included prior to observing data. 

**INCOMPLETE**
Distribution in a Bayesian model 
The prior distribution $\pi(\theta)$
The joint distribution of $X, y$ $$p(X, y)=$$
The prior predictive distribution 
The posterior distribution 
The predictive distribution

In this example the posterior distribtuion corresponds to a beta distribution. 

Two distributions are conjugate if the prior and posterior follow the same distributions; in this case the prior is called the *conjugate prior*. 

The previous method is a **fully bayesian** approach. Sometimes you don't need to fully compute the posterior, it's enough to know the posterior up to a constant (i.e. you don't calculate the denominator in Bayes). In which case we can estimate $\theta$ by the MAP (maximum A posteriori), or by the MLE (maximum likelihood estimator). 

### Bayesian networks 

**INCOMPLETE**
A Bayesian network over random variables $X_1,..., X_n$ consists of 
* a directed acyclic graph (DAG)
* A set of local conditional distributions 

any DAG can be built from 3 structures:
* serial connection 
* diverging connection 
* converging connection 

From the DAG can figure out which variables are conditional/dependent on one another. 

> **Naive Bayes**
> * predicting the value of a categorical variable, $Y$ from a set of features $X_1,..., X_k$

**Generative vs. Discriminative models**

*Generative* - learn $p(x,y)$ from data. Compute $p(y|x)$ using Baye's rule. 
* Examples: Naive Bayes, Bayesian networks in general, ...
* Advantages: Can be used to generate synthetic data 
* Note: Higher asymptotic error but reached more quickly - so can be better for small datasets

*Discriminative models* - Estimate $p(y|x)$ directly from the data
* Examples: Logistic rgression, NNs, ...
* Note: Lower asymptotic error but reached more slowly 

Graphical DAG notation:
* Plate notation - notation for drawing graphical DAGs that collapses the iterations. 
* Grey coloured variables are ones which we observe.
* Variables with a circle around them are modelled as random variables. 

Using Monte Carlo can be a good first way to understand what your posterior is. 

> **Factor analysis model** 
> * In general, latent variable models are regarded as probabilistic models where some of the variables cannot be observed. 
> * Factor analysis summarises a high dimensional observation X of correlated variables by a smaller set of factors Z which are assumed independent a priori. 

### Inference in Bayesian Models 

Have variables $X=(X_1,..., X_n)$. And evidence $X_E$. 

**Inference Methods**

*Exact*

* Brute force: compute $P(X, X_E)$ and maginalise out $X \setminus X_I$.  
* Take advantage of the network structure, and do variable elimination. 

*Inference*

* Sampling 
* Deterministic

**Monte Carlo inference algorithms**

* If the entire population was available, the inference problem could be solved exactly, basically just count the cases. 
* In reality we never have all the data. 
* Monte Cralo operates by drawing an artificial sample from it using some random mechanism. 
* The sample is used to estimate each variable of interest. 
* Two main methods: Importance sampling and MCMC. 

**Importance sampling**

* Fundamental idea is to express the computation we want to solve in terms of the expectation of the random variable.  
* We sample for a different population, $p^*$. $p^*$ is chosen to be very simple, it can be anyhting as long as it is non-negative for the same values as $p$. The accuracy depends on the choice of $p^*$ - if we choose $p^*$ as something close to $p$ then we get better accuracy. 

**MCMC**

* Danger is that you get stuck in one part of the population (local optima). 
* More notes on the GitHub. 

