# Simulation-based inference - 16/6/22
## Henri Pesonen 

Use simulators for real world phenomena. The complexity of the simulators prohibits access to likelihood function. 

**Example: Transmissions of bacterial infections in daycare centers**
* Cross sectional data from SIS model 
* Continuous time model. 
* Want to simulate which child in which daycare carries which strain of data? 

**Example: Personalised medicine**
* Model how cancer cells are evolving in the tissue. 
* Want to simulate how the combination of drugs is kiling the cancerous cells. 

### Simulator 

A computer program defined as $x \approx p(x| \theta)$ that has 
* input paramters 
* stochastic output 

The data can be in any format, i.e. single time series, images, distributions of data points 

### Inference 

Observe the data and infer the values of the paramters that generated them. 

Might use, likelihood, or a bayesian approach, or MLE. 

### Inference without likelihood 

* use the capability to draw simulated data conditioned on the input parameters. 
* likely true paramters values are thought to produce data that are similar enough to the observed data 

**Distance metric**
* Acceptable region defined by distance metric 
*Choice of metric depends on the data format, e.g. could use Eulcidean or L1 etc 

**Data dimensionality**

* In high dimensional spaces it becomes hadrer to generate data close to the observed data. 
* Current approach is to use summary statistics $S$

**How do we select summary statistics?**
* open problem 
* use bespoke summary statistics - could use *domain expertise*, explore the simulator prior to inference, diagnose the inference results. 
* also automatic algorithms for selecting/constructing summary statistics. 

----

Putting this together get: **Rejection Approximate Bayesian Computation (ABC)** 

for $i=1,...,N$, 
$\theta^* ~ p(\theta)$
INCOMPLETE

**Alternative approach**

instead of choosing a fixed threshold can sample a large artifical set and choose a fraction of samples which are most similar to the observed data. Threshold can then be calculated based on which samples were selected. 

Rejection ABC uses samples from the prior. 

* unless we have lots of prior information about the parameters this is difficult 

INCOMPLETE

### Issues to be aware of 

Summary statistics may not catch relevant features of the data
* can decrease dimension too much thus losing information
* didn't decrease dimension enough, so didn't solve the problem 
* can be correlated - have redundant dimensions 

Multiple summary statistics can have widely different scales, so unequal contributions to the distance metric 

Book on this subject: [Handbook of approximate bayesian computation](https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9780367733728)

### Surrogate models

Alternative apporach is to construct surrogate models of parts o

Example: construct approximate likelihood at an abritrary paramter value 

Synthetic likelihood is hardly efficient - the surrogate is fitted at each parameter value seperately. 

**BOLFI - bayesian optimisation for likelihood free inference.** 

This approach uses Gaussian processes. 

Gaussian process surrogates can utilise active learning. 
* Different strategies for selecting parameter values where to query the simulator. This reduces the number of queries to produce reasonable approximations to posterior ditribution. 

Want to find parameter values that minimise the discrepancy function. Uses black box optimisation. Aquisition strategies balance exploration and exploitation. 

Minimising the distance may not be optimal. Want to choose query points that are most informative about the model. 

How to minimise the distance? 
**LCBSC** 

Lower confidence bound selection criteria for minimising the distance. 

**MaxVar** - the maximum variance aquisition method (?) 

or **RandMaxVar** - randomised version. 

Most efficient method: **ExpIntVar** - the expected integrated variance. The drawback of this method is that it is quite difficult to calculate. 

Sampling from surrogate 
* to represent the posterior distribution we require a sample drawn from it. Can use MCMC (?)

### After inference

**How reliable are the results?**
Different error sources:
* algorithm performance 
* model performance 
* simulator performance 

Likelihood free inference methods are based on several levels of approximations - these all add to the total error. 

----

Tutorial using ELFI is [here](https://colab.research.google.com/drive/1Dg9FZe07DJdGw5tZI5PIxNuAuszULsNP?usp=sharing#scrollTo=Vr_EBUiEZjkC)

The generative model is described as a DAG.  

When doing BOLFI, doing NUTS may not be the best choice. You only need to choose the next point to evaluate at, so may as well just use metropolis hastings. 

NUTS sampler is default for the posterior sampler phase. 