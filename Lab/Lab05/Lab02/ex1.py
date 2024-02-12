import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

distrib1 =  np.random.exponential(scale=1/4, size=10000)
distrib2 =  np.random.exponential(scale=1/6, size=10000)
timp1 = 0.4 * distrib1
timp2 = 0.6 * distrib2
final_distribution = 0.4*timp1 + 0.6*timp2
az.plot_posterior({'final_distribution':final_distribution}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()
