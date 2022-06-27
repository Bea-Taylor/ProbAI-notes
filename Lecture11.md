# Bayesian Neural Networks - 17/6/22

### Bayesian Neural Networks 101

Bayesian solution: 
* Put a prior $p(\theta)$ on network parameters $\theta$, e.g. gaussian prior
* approximate bayesian predictive inference: 
* monte carlo approximation 

*Steps for approximate inference in BNNs*
1. construct the $q(\theta) \approx p(\theta|D)$ distribution. 
2. Fit the $q(\theta)$ distribtuion. e.g. using variational inference. 
3. Compute prediction with Monte Carlo predictions 

### Part 1: Basics 
the true posterior: $p(\theta | D) =p(D| \theta)p(\theta)/p(D)$

have an approximate posterior: $q(\theta)$

Use Kullback-Leibler Divergence (KL). (note: when p=q, KL=0, otherwise KL>0.) 

We want to minimise $KL[q(\theta)||p(\theta|D)]=logp(D)-E_{q}[log \frac{p(\theta, D)}{q(\theta)}$

So minimising KL, is eqivalent to mkaximising $L=E_{q}[log \frac{p(\theta, D)}{q(\theta)}$. This is the veidence lower bound (ELBO). 

Can re-write the ELBO: $L=E_{q_{\psi(\theta)}}log p(\theta, D) - KL[q_{\psi}(\theta)||p(\theta)]$. 

___

**Using other $q$ distributions**

Can use more complicated q distributions. 
* Pro: more fleixble approximations -> better posterior approximations 
* Con: higher time and space complexities

Can use *last-layer BNN*. 
* use deterministic layers for all but the last layer
* for the last layer use full covariance gaussian approximate posterior. 

* For regression this is equivakent to bayesian linear regression (BLR) with NN-based non-linear features. We use a KL regulariser for the last layer only. 

Can make it more economic using *MC-dropout*.
* add drop out layers to the network 
* perform drop out during training 

### Case study 1: Bayesian optimisation (BO)

first idea: fit a surrogate function, $f_{\theta} \approx f_0$. 

So have $y_i \approx f_0(x_i)+ \epsilon$

Idea of BO: iterate the following steps
- fit a surrogate function $f_{\theta}$ with uncertainty estimates. 
- use a surrogate function to guide the dataset collection process. 

Start from a samll amount of observations and update the dataset using $D=D \cup \{ (x_*, y_*)\}$. 

Can use the upper confidence bound (UCB) as an acquisition function. 

### Case study 2: Detecting adversarial examples 

Hypothesis:
* adversarial examples 

**Uncertainty measures**
total uncertainty = epsitemic uncertainty + aleatoric uncertainty 

imagine flipping a coin
*epistemic uncertainty* how much do i believe the coin is fair?
*aleatoric uncertainty* what's the next coin flip outcome? 

Can use shannon entropy to compute uncertainty. 
$H[p]=- \sum plog(p)$ 

### Applications 
u
BBNs in medical imaging - Super resolution 
