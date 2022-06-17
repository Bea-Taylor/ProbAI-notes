# Human-centered ML - 16/6/22
## Fani Deligianni 

[Computing technologies for healthcare team](https://www.gla.ac.uk/schools/computing/research/researchthemes/healthcaretechnologies/)

**Why do we need human-centered AI?**

* acountability 
* technical robustness and safety 
* oversight 
* privacy and data governance 
* non discrimination and fairness
* transparency 
* soietal and enviromental wellbeing 

Healthcare models - how do we go from ML models to somehting which can be used [here](https://academic.oup.com/ckj/article/14/1/49/6000246)

**ABCD Guide** 
* A: callibration in the large (this means external validation), or the model intercept
* B: calibration slope
* C: discrimination with the reciever operating characteristic curve 
* D: clinical usefulness with decision curve analysis

Is risk under or over estimated? Look at predicted risk vs. observed proportion. 

Think about clinical consequences, what matters more sensitivity or specificity? 

i.e. in breast cancer a false-negative is more harmful than a false-positive. 

Would need to do decision analysis - explicit validation of health outcomes. Can look at net- benefit. 

**Net benefit**

$net \space benefit = sensitivity*prevalence-(1-specificity)*(1-prevalence)* \frac{threshold probability}{1-threshold probability}$

Compare the model with treating all positive, treating none or treating based on another factor, i.e. duration of illness. 

Experts can inform the threshold for treating people. 

**Uncertainty**

Frequentist approach provides confidence intervals. 

**WHat guarantees can we use against dicriminatory bias?**
* Look at calibration within groups.

### Transparency and explainability

* explainability is required to ensure impartial decision making process
* people have a right to know wy decisions were made for them. 

INCOMPLETE
**Who is the target audience of explainability?**
* clinicians - need to ttrust the model 
* patients - need to understand decisions 
* data scientists/developers ...

Traditionally use interpretable models, need to make an efoort to explain more complicated models. i.e. NN can be explainable models. 

**How do we explain a model?** 
* Think about local (personal decision) based explanations vs. more global explanations


Paper on: [The next generation of medical decision support](https://pubmed.ncbi.nlm.nih.gov/33733193/)