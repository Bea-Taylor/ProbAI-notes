# Normalising Flows - 15/6/22
## Didrik Nielsen 

Lectures slides [here](https://github.com/probabilisticai/probai-2022/tree/main/day_3/3_didrik)

Normailising flows describe the change of probably density through a series of invertible maps. 

----

Q: Where are flows useful?

A: anywhere you need a flexible density $p(x)$. 

----

Q: Why flows?

A: They can exactly get the density estimate, and it's quite fast. Sampling is also fast. 

----

Useful for generative modelling and also for variational inference. 

### The framework 
To construct $p(x)$ the key things you need are:
* Base distribution $p(z)$
* and the mapping $f$

we require $f$ to be invertible/bijective so that we can map $x$ to $z$, and $z$ to $x$. 

Can use it for sampling:
$z \approx p(z)$
$x=f(z)$

and can also use it for density: 
$log p(x) =log p(z) + log|det \frac{\partial z}{\partial x}|$
$z=f^{-1}(x)$

The important parts are:
* *Forwards*: $x=f(z)$
* *Inverse*: $z=f^{-1}(x)$
* *The jacobian determinant*: $det \frac{\partial z}{\partial x}$

Deriving the change of variable formula: 
In one dimension: $p(x)=p(z)|\frac{dz}{dx}|
In higher dimensions: $p(x)=p(z)|det \frac{\partial z}{\partial x}|$

Need to structure the flow to make the jacobian easier to compute. 

Can stack multiple bijections. The composiiton of bijections is a bijection. 

### Coupling flows 

**How to build efficient flows?**

All about developing layers that: 
* are expressive 
* are invertible 
* have chea to compute jacobian determinants. 

Main categories of flows are: 
* Det. identities
* Autoregressive 
* Coupling - the most popular type of flow, fast in both directions and relatively easy to compute
* unbiased

**Coupling flows**

Don't/idenity transform the first d

Forward:
$x_{1:d}=z_{1:d}$
$x_{d+1:D}=\exp^{-\alpha_{d+1:D}} \dot (z_{d+1:D} - \mu_{d+1:D})$

Inverse:
$z_{1:d}=x_{1:d}$

$z_{d+1:D}=x_{d+1:D} \dot e^{\alpha_{d+1:D}}+\mu_{d+1:D}$

Attempted to implement this in python, notebook for this is [here](https://github.com/probabilisticai/probai-2021/blob/main/Day5/2_nielsen/realnvp_solution.ipynb)

This was a very simple example, curent work is on photographs.

### Autoregressive flows

Autoregressive flows are either fast for sampling or fast for density - but not both. 

If you decide to use autoregreesive flow then need to use autoregressive NNs. 

Can use *masked autoregressive NNs*. For example: MADE (for vectors), WaveNet (for sequences) or PixelCNN (for images). 

### Continuous-time flows

[Neural ODE](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)

$\frac{\partial z (t)}{\partial t} = f_{\theta}(z(t),t)$

Uses an instantaneous change of variables formula.

Related to [score based diffsuion models](https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html)

### Residual flows

$z = f(x) = x + h(x)$ for $h$ - contractive function (lipshitz cts?). 

Use [hutchinson's trace estimator](http://blog.shakirm.com/2015/09/machine-learning-trick-of-the-day-3-hutchinsons-trick/) to calculate the jacobian. 

They truncate the sum in hutchinsons trace estimaor. This introduces bias - which they get round this by sawpping truncation for something called the russian roulette estimator. 

### Discrete flows

It's hard to transform a discrete distribution. 

[Discrete flows](https://proceedings.neurips.cc/paper/2019/hash/e046ede63264b10130007afca077877f-Abstract.html): 
$z_d = (\mu_d + \sigma_d x_d) mod K$

[Integer Discrete flows](https://proceedings.neurips.cc/paper/2019/hash/9e9a30b74c49d07d8150c8c83b1ccf07-Abstract.html): $z_d =  x_d + \mu_d$

Both of these papers used a *straight through estimator* 

### Surjective flows 

[surVAE Flows](https://proceedings.neurips.cc/paper/2020/hash/9578a63fbe545bd82cc5bbe749636af1-Abstract.html)

* Authored by the lecturer!

Can consider flows which are no longer bijective but instead surjective. 

Examples include: 
* $x = round(z)$
* $x=z[:n]$ 'tensor slicing'
* $x=argmax(z)$

## Bonus: Variational Inference with flows
Maximising the likelihood is exactly the same as minimising the KL. 

Variational inference 

Learn posterior approximation: 

$q_{\lambda}(\theta) \approx p(\theta | D)$ 

*Mean field Gaussian*: The common choice (which is traditional/'old-fashioned') is to use the mean field approximation, choosing each distribution to be Normal. 

We can adapt this, so where we would use a mean field approximation we can instead use a flow. 

---
Q: What kind of flows do we want to use for variational inference?

A: Well, desired properties would be : fast sampling, density for generated samples. 