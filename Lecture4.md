# Variational Inference and Optimization 1 - 14/6/22
### Helge Langseth, Andres Masegosa, Thomas Nielsen 

## Part 1 (Before lunch)
link to GitHub [here](https://github.com/PGM-Lab/2022-ProbAI/tree/main/Day2-BeforeLunch)

Idea of variational inference: approximate $p(\theta | D)$ using $q(\theta | D)$. Note: as shorthand write $q(\theta) = q(\theta | D)$. Then do  abunch of maths, so that finding $q(\theta)$ is an optimisation problem. 

**Some terminology**
* KL - Kullback Leibler divergence 
* MF - mean field assumption 
* CAVI - coordinate ascent variational inference algorithm 

So want to minimise 
$ \hat{q}= argmin \delta (q(\theta) || p(\theta | D)) $. 

Note: ideally $\hat{q}$ would be a metric. 

**INCOMPLETE**

### Solving the variational Baye's optimisation 

ELBO: $L(q) = E_q [log \frac{p(\theta, D)}{q(\theta)}]$

**INCOMPLETE**

After some maths (given in the presenttaion/notes) we get: 

$L(q) = c - KL()$ **INCOMPLETE**

**Two main assumptions**

* *Mean field*: assume $q(\theta)=\prod_i q_i(\theta_i)$ ...
* we optimise with respect to $q_j()$ and keep others fixed - i.e. do coordinate ascent. 

**Setup** 
* Have observed the data $D$, and can calculate the full joint, $p(\theta, D)$. 
* We use the ELBO as our objective and assume that $q(\theta)$ factorises. 
* We posit a variational family of distributions $q_j (\theta_j |\lambda_j)$, i.e. we choose the distributional form, while wanting to optimise the parameterisation $\lambda_j$. 

The variational updating rules are guaranteed to never decrease the ELBO - i.e it should always monotonically increase. If it's not monotonically increasing then you know you've gone wrong!!! 

### Example: a simple Gaussian model 
See the jupyter notebook [here](https://github.com/PGM-Lab/2022-ProbAI/blob/main/Day2-BeforeLunch/notebooks/students_simple_model.ipynb)

### Real data example 

Going to look at dataset which compares a countries terrain ruggedness to GDP, and how is this effected by whether the nation is in Africa or not? 

## Part 2 (After lunch :) )

So far we have mostly considered linear models BUT a linear model might not be the right choice - so perhaps should consider a neural network. 

It's quite easy to fit a curve using a neural network - but we only get one solution. Instead, we want to a Bayesian neural network. 

### Stochastic gradient descent 

We want to optimise the ELBO using gradient descent. 

> gradient ascent algorithm for maximising a function $f (\lambda)$
> * initialise $\lambda^{(0)} 
> * then update according to 

Standard gradient ascent is not enough for ELBO optimisation 
We won't be able to calculate the gradient exactly simce:
* we may have to resort to mini-batching ( calculating the gradient for a 'random subset')
* even for a mini batch may not be able to calculate the gradient exactly. 

**Black box variational inference (BBVI)**
Idea: cast inference as an optimisation problem. 

gardient ascent uses the gradient (derivative) to update the values. I.e. we take a sample, then update using the gradient ascent, and this will narrow into the desired solution. (It's sort of comparable to the sampling and updating step in MCMC). 

**Score fucntion gradient**

The [notebook](https://github.com/PGM-Lab/2022-ProbAI) 'students_BBVI' gives a really useful example of this. 

**Reparameterised gradient**

Can look at re-parameterising the distributions. This only works for cts distributions since $log p(\theta, D)$ need to be differentiable wrt $\theta$. 

Interesting paper on this topic:
[Advances in variational research](Advances in variational inference.")

**Score function gradient vs. reparameterised gradient**

*Score function gradient*: the gradient goes towards the mode of q. 

*Reparameterisation gradient*: the gradient goes approximately towards the mode of p. 

The reparametrisation trick is really good IF you have a function which can be reparameterised - not every function can be! 
The reparametrisation trick doesn't use the mean field approximation. 

**Summary of reprameterisation**
Reparameterisation: gradients align with model's gradient. But: 
* requires $q(\theta | \lambda)$ to be reparametrisable. 
* requires $ln p(D, \theta)$ and $ln q(\theta | \lambda)$ be differentiable - i.e. no categorical variables. 

**Automatic variational inference in PPLs**

How does this actually work in PPLs? We looked at a very manual example, in reality much of this is already encoded in Pyro. 

* *Manual*: define your data and the model 
* *Manual/automatic*: define the variational distribution
* *Automatic*: optimise the ELBO

### Pyro models 

Pyro models: 
* random variables `pyro.samples`
* observations `pyro.samples` with the `obs` argument 

Inference problem:

$p(temp | sensor=18)$

Variational solution:

$min KL( q(temp)|| p(temp | sensor=18))$

Pyro guides:
* guides are abitrary stochastic functions. 
* guides produce samples for those variables of the model which are not observed. 

```python 
# the observations
obs = ('sensor': torch.tensor(18.0))

def model(obs):
    # this is an initialisation value for the temp
    temp = pyro.sample('temp', dist.Normal(15.0, 2.0))
    sensor = pyro.sample('sensor', dist.Normal(temp, 1.0), obs=obs['sensor'])
```

```python 
# the guide
def guide(obs):
    a = pyro.param('mean', torch.tensor(0.0))
    b = pyro.param('mean', torch.tensor(1.0), constraint=constraints.positive)
    temp = pyro.sampe('temp', dist.Normal(a,b))
```

Note: signatures should be the same for the model and guide. 

Example of variational inference using Pyro [here](https://github.com/PGM-Lab/2022-ProbAI/blob/main/Day2-AfterLunch/notebooks/solution_simple_gaussian_model_pyro.ipynb)

Exmple of using Bayesian linear regression using Pyro [here](https://github.com/PGM-Lab/2022-ProbAI/blob/main/Day2-AfterLunch/notebooks/bayesian_linear_regression.ipynb)

Example of using Bayesian logistic regression using Pyro [here](https://github.com/PGM-Lab/2022-ProbAI/blob/main/Day2-AfterLunch/notebooks/students_bayesian_logistic_regression.ipynb)

## Part 3 (After the coffee break)

**The variational auto encoder (VAE)**

Introduce a function that maps a data point to a variational parameter. This function is an ENCODER NETWORK. 

## Recap and conclusions 

**INCOMPLETE** 

Variational inference 

* Provides a distribution approximation to $p$
* Mean-field is a divide and conquer strategy for  high dimensional posteriors 
* main caveat - $q(\theta |\lambda)$ tends to underestimate the uncertainty of $p(\theta | D)$

Coordinate ascent variational inference 

* analytical expressions for some models
* CAVI is very efficient and stabel if it can be used
* In principle requires manual derivation of updating equations - but can use toolboxes like Pyro

Gradient-based variational inference 

* provides the tools for VI over arbitrary probabilistic models 
* directly integrates with the tools for deep learning 

Probabilistic programming languages 

* PPLs fuel the 'build -  compute - critique - repeat' cycle 
* these offer ease and flexibility of modelling 

What's next?

* The 'VI toolbox' is reaching maturity 
* no longer a research area, more a pre-requisite for probabilistic AI 
