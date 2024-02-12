import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

np.random.seed(42)
miu = 10
sigma = 2
timpimedii_asteptare = np.random.normal(miu, sigma, 200)


with pm.Model() as model:
    
    miu_prior = pm.Normal('miu', mu=10, sd=5) #dsitrib a apriori --> miu
    
    
    sigma_prior = pm.HalfNormal('sigma', sd=5) #distrib. apriori --> sigma
    
    
    observed_data = pm.Normal('observed_data', mu=miu_prior, sd=sigma_prior, observed = timpimedii_asteptare) # pe baza distrib normala am o distrib obs.

    with model:
        trace = pm.sample(1000, tune=1000)


