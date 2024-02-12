import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
#Ex1

#aici,am generat datele
dummy_data = np.loadtxt('./data/dummy.csv', delimiter=',', skiprows=1)
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

order_1a = 5 #am schimbat cum s-a cerut la pct 1 a order la 5
x_1p_1a = np.vstack([x_1**i for i in range(1, order_1a+1)])
x_1s_1a = (x_1p_1a - x_1p_1a.mean(axis=1, keepdims=True)) / x_1p_1a.std(axis=1, keepdims=True)
y_1s_1a = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_p_1a:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=10, shape=order_1a)
    epsilon = pm.HalfCauchy('epsilon', beta=1)
    mu = alpha + pm.math.dot(beta, x_1s_1a)
    y_pred_1a = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s_1a)
    trace_p_1a = pm.sample(2000, tune=1000)

az.plot_posterior(trace_p_1a)
plt.show()

#Pct b)
with pm.Model() as model_p_beta_100:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=100, shape=order_1a) #sd = 100 si sd - np.array[....]
    epsilon = pm.HalfCauchy('epsilon', beta=1)
    mu = alpha + pm.math.dot(beta, x_1s_1a)
    y_pred_beta_100 = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s_1a)
    trace_p_beta_100 = pm.sample(2000, tune=1000)

with pm.Model() as model_p_beta_array:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order_1a) # sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
    epsilon = pm.HalfCauchy('epsilon', beta=1)
    mu = alpha + pm.math.dot(beta, x_1s_1a)
    y_pred_beta_array = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_1s_1a)
    trace_p_beta_array = pm.sample(2000, tune=1000)

az.plot_posterior(trace_p_beta_100)
plt.show()

az.plot_posterior(trace_p_beta_array)
plt.show()

#Ex2
num_points_2 = 500 #500 pct
x_2 = np.linspace(x_1.min(), x_1.max(), num_points_2)
y_2 = 3 * x_2**2 + np.random.normal(scale=10, size=num_points_2) #polyn
order_2 = 5 #order 5
x_2p = np.vstack([x_2**i for i in range(1, order_2+1)])
x_2s = (x_2p - x_2p.mean(axis=1, keepdims=True)) / x_2p.std(axis=1, keepdims=True)
y_2s = (y_2 - y_2.mean()) / y_2.std()

with pm.Model() as model_p_2:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=10, shape=order_2)
    epsilon = pm.HalfCauchy('epsilon', beta=1)
    mu = alpha + pm.math.dot(beta, x_2s)
    y_pred_2 = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_2s)
    trace_p_2 = pm.sample(2000, tune=1000)

az.plot_posterior(trace_p_2)
plt.show()

#Ex3
order_3 = 3
x_3p = np.vstack([x_1**i for i in range(1, order_3+1)])
x_3s = (x_3p - x_3p.mean(axis=1, keepdims=True)) / x_3p.std(axis=1, keepdims=True)
y_3s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_p_3:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=10, shape=order_3)
    epsilon = pm.HalfCauchy('epsilon', beta=1)
    mu = alpha + pm.math.dot(beta, x_3s)
    y_pred_3 = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y_3s)
    trace_p_3 = pm.sample(2000, tune=1000)

az.plot_posterior(trace_p_3)
plt.show()