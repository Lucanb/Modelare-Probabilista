import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

np.random.seed(42)
miu = 10
sigma = 2
timpimedii_asteptare = np.random.normal(miu, sigma, 200)


with pm.Model() as model:
   
    miu_prior = pm.Normal('miu', mu=10, sd=5)  #distrib a priori pt miu miu
    
    #distribuție apriori pentru sigma
    sigma_prior = pm.HalfNormal('sigma', sd=5)
    
    
    observed_data = pm.Normal('observed_data', mu=miu_prior, sd=sigma_prior, observed = timpimedii_asteptare) #ma bazez pe distrib normala so am observed_data

    with model:
        trace = pm.sample(1000, tune=1000)


pm.plot_posterior(trace, varnames=['sigma'], credible_interval=0.95) #vizualizare grafică a distribuției a posteriori pt sigma
plt.title('Distribuția a Posteriori pentru Sigma')
plt.show()

