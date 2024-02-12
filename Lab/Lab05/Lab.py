import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import csv

f = open("trafic.csv", "r")
d = csv.reader(f)
h = next(d)
r = []
for row in d:
    r.append(int(row[1]))

t = np.array(r)
n = len(t)

with pm.Model() as m:
    a = 1.0 / t.mean()
    l1 = pm.Exponential("l1", a)
    l2 = pm.Exponential("l2", a)
    l3 = pm.Exponential("l3", a)
    l4 = pm.Exponential("l4", a)
    l5 = pm.Exponential("l5", a)

tau1 = (7 - 4) * 60
tau2 = (8 - 4) * 60
tau3 = (16 - 4) * 60
tau4 = (19 - 4) * 60

with m:
    idx = np.arange(1200)
    lam = pm.math.switch(tau1 < idx, l1, pm.math.switch(tau2 < idx, l2,
                                                    pm.math.switch(tau3 < idx, l3,
                                                      pm.math.switch(tau4 < idx, l4, l5))))

with m:
    o = pm.Poisson("obs", lam, observed=t)

with m:
    s = pm.Metropolis()
    tr = pm.sample(100, tune=100, step=s, return_inferencedata=False, cores=1)

l1s = tr['l1']
l2s = tr['l2']
l3s = tr['l3']
l4s = tr['l4']
l5s = tr['l5']

print("Medie l1:", l1s.mean())
print("Medie l2:", l2s.mean())
print("Medie l3:", l3s.mean())
print("Medie l4:", l4s.mean())
print("Medie l5:", l5s.mean())

az.plot_posterior({'l1': l1s, 'l2': l2s, 'l3': l3s, 'l4': l4s, 'l5': l5s})
plt.show()
