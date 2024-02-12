import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

yValues = [0,5,10]
thetaValues = [0.2,0.5]


for y in yValues:
    for theta in thetaValues:
        with pm.Model() as model:
            n = pm.Poisson("n",mu=10)
            observedVal = pm.Binomial(f"observed values:{y}{theta}", n=n, p=theta, observed=y)
            sample = pm.sample(1000)

            az.plot_posterior(
                sample,
                var_names=["n"],
                point_estimate="mean",
            )
            plt.title(f"Y = {y}, theta = {theta}")


plt.show()