import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

df = pd.read_csv("Admission.csv")

scor_gre = df['GRE'].values
medie_gpa = df['GPA'].values
rezultate_admitere = df['Admission'].values

#aici am luat modelul logistic
with pm.Model() as model_logistic:
    # A priori slab informativ pentru parametrii
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)

    pi = pm.math.invlogit(beta0 + beta1 * scor_gre + beta2 * medie_gpa)
    admitere_likelihood = pm.Bernoulli('admitere_likelihood', p=pi, observed=rezultate_admitere)

    idata = pm.sample(5000, tune=1000, chains=2)

# Analiza rezultatelor
az.plot_posterior(idata, var_names=['beta0', 'beta1', 'beta2'], hdi_prob=0.94)
plt.show()

# Granița de decizie
beta0_amostra = idata['beta0']
beta1_amostra = idata['beta1']
beta2_amostra = idata['beta2']

granița_decizie_medie = -beta0_amostra.mean() / beta2_amostra.mean()

#Intervalul HDI pt granita de dec.
hdi_granița_decizie = az.hdi(-beta0_amostra / beta2_amostra, hdi_prob=0.94)

print(f"Granița de decizie în medie: {granița_decizie_medie}")
print(f"Intervalul HDI pentru granița de decizie: {hdi_granița_decizie}")

# Cu scorul aferent GPA,GRE am fct intervalul 90 HDI
date_student_1 = {'GRE': 550, 'GPA': 3.5}

with model_logistic:
    prob_admitere_student_1 = pm.math.invlogit(beta0 + beta1 * date_student_1['GRE'] + beta2 * date_student_1['GPA'])
    hdi_probabilitate_student_1 = pm.hdi(prob_admitere_student_1, hdi_prob=0.9)

print(f"Intervalul HDI pentru probabilitatea de admitere pentru studentul 1: {hdi_probabilitate_student_1}")
