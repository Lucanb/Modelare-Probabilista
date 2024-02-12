import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import pgmpy

np.random.seed(1)

exp = np.random.exponential(scale=1/4, size=10000)

# Sample from the gamma distributions
distrib1 = stats.gamma.rvs(4, scale=1/3, size=10000) + exp
distrib2 = stats.gamma.rvs(4, scale=1/2, size=10000) + exp
distrib3 = stats.gamma.rvs(5, scale=1/2, size=10000) + exp
distrib4 = stats.gamma.rvs(5, scale=1/3, size=10000) + exp

k = 0
final_distribution = 0.25 * distrib1 + 0.25 * distrib2 + 0.3 * distrib3 + 0.2 * distrib4
for i in final_distribution:
    if i > 3:
        k += 1

prob = k / 10000
print(prob)
az.plot_posterior({'final_distribution': final_distribution})
plt.show()