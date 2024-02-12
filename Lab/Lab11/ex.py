import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

#incepem generarea pe 500 date
clusters = 3
num_cluster = [200, 150, 150]
n_total = sum(num_cluster)
means = [5, 0, 3]
stDev = [2, 2, 2]
mixture = np.random.normal(np.repeat(means, num_cluster),
                       np.repeat(stDev, num_cluster))
az.plot_kde(np.array(mixture))
plt.show()
#aici incepem pe 2-3-4 sa calibr. mixturile
with pm.Model() as model1:
    p1 = pm.Dirichlet('p1', a=np.ones(2))
    mean1 = pm.Normal('mean1', mu=np.array(mixture).mean(), sigma=10, shape=2)
    sd1 = pm.HalfNormal('sigma1', sigma=10)
    y1 = pm.NormalMixture('y1', w=p1, mu=mean1, sigma=sd1, observed=np.array(mixture))
    data_sample1 = pm.sample(random_seed=123, return_inferencedata=True, cores=1, idata_kwargs={'log_likelihood': True})

with pm.Model() as model2:
    p2 = pm.Dirichlet('p2', a=np.ones(3))
    mean2 = pm.Normal('mean2', mu=np.array(mixture).mean(), sigma=10, shape=3)
    sd2 = pm.HalfNormal('sigma2', sigma=10)
    y2 = pm.NormalMixture('y2', w=p2, mu=mean2, sigma=sd2, observed=np.array(mixture))
    data_sample2 = pm.sample(random_seed=123, return_inferencedata=True, cores=1, idata_kwargs={'log_likelihood': True})

with pm.Model() as model3:
    p3 = pm.Dirichlet('p3', a=np.ones(4))
    mean3 = pm.Normal('mean3', mu=np.array(mixture).mean(), sigma=10, shape=4)
    sd3 = pm.HalfNormal('sigma3', sigma=10)
    y3 = pm.NormalMixture('y3', w=p3, mu=mean3, sigma=sd3, observed=np.array(mixture))
    data_sample3 = pm.sample(random_seed=123, return_inferencedata=True, cores=1, idata_kwargs={'log_likelihood': True})
#aici fa compararea cu metodele waic si loo
loocvScore = az.compare({'sample1': data_sample1, 'sample2': data_sample2, 'sample3': data_sample3},
                       method='stacking', ic='loo', scale='deviance')

waicScore = az.compare({'sample1': data_sample1, 'sample2': data_sample2, 'sample3': data_sample3},
                      method='stacking', ic='waic', scale='deviance')
print(loocvScore)
print(waicScore)