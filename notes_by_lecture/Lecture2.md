# Probabalistic Modeling and Programming - 13/6/22
## Andres R. Masegosa, Thomas D. Nielsen 

Scikit learn [cheat sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) for choosing a model

PPL = Probabilistic programing language 

Q: Why PPLs?

A: 
* stacked architecture
* simplify probabilistic machine learning code
* reduce development time and cost to encourage experimentation
* reduce the necessary level of expertise
* democratisation of the development of probabilistic ML systems 

** Brief history of PPLs**

90s/early 2000s - 1st generation of PPLs
* turing complete probabilistic programming languages
* Used monte carlo methods
* didn't scael to big data, or higher dimensional data 

2nd generation of PPLs
* inference engine based on message passage algorithms 
* scaled to large datsets and high dimensional data 
* limited probabilistic model family 

3rd generation of PPLs - where we are now 
* pyTorch, Pyro, PyMC etc...
* black box variational inference and hamiltonian monte carlo 

***Practise***
See jupyter notebook [here](https://github.com/PGM-Lab/2022-ProbAI)

Pyro allows you to visualise the model, so that you can see dependencies `pyro.render_model(model)` 

In the ice-cream shop model: 
* N - number of data points
* $\alpha, \beta$ are coefficients for the linear desciption of how ice-cream sales depend on temp. They are random variables. 
* $\mu$ is the mean about which temperature is distributed. it is a random variable. 
* $t_i$ is the actaully temperature on a given day $i$, it is distributed normally about the mean $\mu$. 
* $s_{t,i}$ is the observed temperature as observed from a sensor. It is dependent on the temperature. It is distributed normally about the actual temperature with a std deviation of 1 to account for variation in the measurement. 
* $s_i$ is the number of sales on day $i$. It is a poisson distribution, depending on the temperature that day. 