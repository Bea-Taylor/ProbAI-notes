# Gaussian Processes 15/6/22
## Arno Solin

### Pragmatic introduction to Gaussian processes

* A random vector is said to have a multivariate Gaussian distru=ibtuion if all linear combinations of x are Gaussian distributed. 

* A gaussian process can be considered as a distribution over functions. 

* A Gaussian process is completely defined by its mean and covariance. 

### Challenges that break the beauty 

**Three main challenges**
* *Scaling to large data*. A naive solution to deal with the expanded covariance amtrix requires $O (n^3)$ compute. 

* *Dealing with non-conjugate likelihoods*. 

* *Representational power*. Gaussian processes are ideal for problems where it is easy to specify meaningful priors. If you can't... it becomes harder. 

*Scaling to large data*
* exploit structure of the data 
* exploit structure of the GP prior 
* solve the linear system approximately - i.e. conjugate gradient solvers 
* split problem into smaller chunks. Split domains, chunk the data, ...
* approximate the problem
* approximate the problem solution (SVGP - sparse variational gaussian processes). 

*Dealing with non-conjugate likelihood models*
* MCMC - accurate but generally heavy 
* Laplace approximation - fast and simple - but old fashioned!
* Expectation approximation - efficient but tricky, requires lots of tuning 
* Variational methods - dominant method today

*Representational power*
* GPs can be seen as shallow but infinitely wide. This might not be the right model for the job!!! i.e would be a bad choice for a low dimensional manifold in a high dimensional space. 
* BUT, they can be good tools to combine with other models! 

### Connections and approaches to GPs

**Connection to Neural Networks**
* showed that untrained single layer NNs converge to GPs in the limit of infinite width. 

**Connection to physics**
* models often written in terms of ODEs/PDEs etc
* GPs used as structured priors. 

**Connection to Bayesian optimisation** 
* used to figure out the next optimal point to look at - could this be used for healthcare decision modelling???

### Recap 

* Gaussian processes provide a plug and play framework for probabilistic inference and learning 
* give an explicit way of understanding prior information in a problem 
* provide meaningful... 

Good (old-ish) book - Gaussian processes for machine learning - Rasmussen. 