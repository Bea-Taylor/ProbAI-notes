# Bayesian Workflow 
## Elizaveta Semenova

* Workthrough on [github](https://github.com/elizavetasemenova/ProbAI-2022)

Posterior = likelihood * prior 

general principles of bayesian inference: 
specify a complete Bayesian model then sample the posterior distribution of the parameter $\theta$. 

PPLs are designed to let the user focus on modelling while inference happens automatically. Users need to specify: prior and likelihood. 

**Diagnosing MCMC outputs**

*Convergence diagnostics* 

* $\hat{R}$
* traceplots 

*Effective sample size*

* in MCMC samples will be autocorelated within a chain 

We use multiple chains and inspect convergence after warm up (disregard burn in period). Ideally we want all the chains to looks stationary and agree with one another. We can also look at the posterior distributions for each chain. 

### Modern Bayesian workflow

Modern Bayesian workflow is quite complicated - Gelman [pdf](http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf)

**Prior predictive checking**

Prior predictive checking consists in simulating data from the priors. 
* visualise priors (especially after transformation) 
* this shows the range of data compatible with the model 
* it helps understand the adequacy of the chosen priors 

**Iterative model building**

**INCOMPLETE**
A possible realisation of the Bayesian workflow loop:
* understand the domain and problem (background knowledge) 


**Example: ODE-based compartmental model (SIR model)**

Do a prior predictive check: calculate a range of possible trajectories for the ODE (so no use of observed data). Then we check whether our observed data lies in this range - if it does this indicates that the model is appropriate, as it's possible that the chosen priors and model give rise to the observed data.  

Might find that our model isn't very good. Perhaps it doesn't pass our prior predictive check. Perhaps there is poor mixing of the chains. Think - what have we missed? In an SIR model of Covid maybe we haven't taken account of under-reporting, or the incubation period. Then perhaps we need to model the control measures? 

**Example: Drug concentration response**

* Concentration response experiments are used to rank drug candidates. 
* Traditionally the drugs yield sigmoidal curves - characterised by a plateau at a high drug concentration. 
* Curves show a loss of effect at higher doses, known as a 'hook effect' 

Domain understanding - what do the lcinical experts effect the model to show? 
Looking to fit a curve that is 
* flat at low concentrations
* is able to capture hook effect

Can describe this using Traditional Hill's model. Maybe this model isn't the best fit? Use a more flexible approach: Gaussian processes allows us a flexible curve shape. 

We look at diagnostics, trace plots, chain and posteriors. But when we look at the posterior it seems a bit over-fitted? 

Can use kernel design - this allows us to specify a wider range of gaussian process priors. 

This is a new model and complicated. To better understand it we run prior predictives - want to understand how the parameter changes the model. 

**Conclusions**
* use domain knowledge (expert guidance on the real-world problem)
* priors informed by domain knowledge
* using external data to point at a latent variable 
* using a more complex model 

No one size fits all, Bayesian workflow is all about updating and considering what we're doing!

Have postdoc positions at Oxford - contact: Seth Flaxman, Elizaveta Semenova :) 