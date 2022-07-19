# Deep Generative Models - 15/6/22
## 

### Variational auto-encoders

[Variational inference a review for statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773)

Assume data was produced by a generative model: p(z|x)=p(x|z)*p. x is our observed data. z is our latent variable which we want to find out. 

Mean field variational inference is one of the simplest approaches. It makes assumptions that $q(z)$ can be wrtitten as a product of distributions over every latent variable. 

In variational inference we're always interested in minimising $KL(q||p)$. The factorised version of the possible posterior assumes all variables are independent, so it will underestimate the realtionship between variables. 

Both $KL(p||q)$ and $KL(q||p)$ are speicifc instances of $\alpha$ divergence. 

**Auto-encoders** - copy nput to output whilst going through a bottleneck. The code in the botleneck should be a compressed reprentation of the data $x$. Can use auto encoders as generative models - they learn representtaions of the data and generate new data.

Variational inference: optimise paramters of $q(z|x)$ for each x seperately. 

Amortised variational inference: model paramters of $q(z|x)$ with a neural network $nn_{\psi}(x)$. 

This paper - [an introduction to variational autoencoders](https://www.nowpublishers.com/article/Details/MAL-056) has some nice details/overview of the reparameterisation trick. The reparameterissation trick rewrites z as a deterministic function which makes it easier to sample. 

The reparametrisation trick relies on you being able to write z as a determinisitc function g. 

Can easily reparametrise for gaussian distribtuions and also for full-covariance gaussian distributions (i.e. where the variables influence one another)

Evidence lower bound might sometimes not be good enough. Then turn to *marginal likelihood estimation: importance sampling*. 

**Advances in VAEs**

* [Ladder VAEs](https://proceedings.neurips.cc/paper/2016/hash/6ae07dcb33ec3b7c814df797cbda0f87-Abstract.html). Have more layers of latent random variables - in practise this isn't always great - so change the order of the dependencies of the random variables.

* [NVAE](https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html). Can be used to produce high resiltuion pictures using autoencoders. 

### Normalising flows

**Normalising flows for variational inference:**

If the true posterior is complicated and $q(\psi)$ is a simple gaussian, I can never get that close to the true posterior. 

I want to transform $q$ so that I end up with something closer to the true posterior. This helps to get correlated distributions - i.e. where variables are dependent. To do this I apply normalising flows. 

Start by sampling $z_0$ from a simple distribution. Then going to apply a sequence of transformations to end up with soemthing more complicated. i.e. then apply a sequence of invertible transformations: $f_k : \R^D \rightarrow \R^D$. 

$z_k=f_k \prod f_{k-1} \prod... \prod f_1(z_0)$ and for each transformation $z_k=f_k(z_{k-1})$. End up with a way of getting $ln q_k (z_k)$ that allows for more complicated relationships between latent variables. 

There are some practical requirements for normailisng flows for variational inference:

* need flexible invertible transformations - as inflexible transformations lead to long sequences of flows for flexible posteriors. By flexible it means we want transformations which can do quite a lot. 
* easily computable jacobian determinants, as we need to know this to calculate $ln q_k (z_k)$. 

Q: What would these transformations look like in practise?

A: [Planar flows](http://proceedings.mlr.press/v37/rezende15.html) 
This is invertible, but not that flexible. However the jacobian determinant is easy to compute - which is where this method really shines. 

**Generative normalising flows**

What about using these alone as a generative model (i.e. without autoencoders)
Associate the final transformed random variable to the observed data. 
Then will take the inverse of the transformations. This leads to an additional practical requirement - the inverse fucntion needs to be easy to compute. 

### Denoising diffusion models 

Idea: take data, assume it corresponds to a random variable. Apply a series of transforamtions to corrupt it into something with noise. Then reverse process - train a probabilistic model which matches the denoising at each step. I.e. at each step I try and make it a little bit less noisy. 