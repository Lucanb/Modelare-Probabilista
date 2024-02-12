import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

model = pm.Model()

with model:
    urgent = pm.Bernoulli('U', 0.05)
    reducere = pm.Bernoulli('R', 0.2)
    cumpara_p = pm.Deterministic('C_p', pm.math.switch(reducere, pm.math.switch(urgent, 1, 0.5), pm.math.switch(urgent, 0.8, 0.2)))
    cumpara = pm.Bernoulli('C', p=cumpara_p, observed=1)

with model:
    trace = pm.sample(20000)

df = trace.to_dataframe(trace)

p_urgent = df[(df['U'] == 1)].shape[0] / df.shape[0]
print(p_urgent)



az.plot_posterior(trace)
plt.show()

