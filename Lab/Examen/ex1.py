import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt


#aici am fct load la date pt punctul a)
Titanic = pd.read_csv('Titanic.csv')


Titanic['Age'].fillna(Titanic['Age'].median(), inplace=True)#Imi lipseau unele valori asa ca am adaugat


Y = Titanic["Survived"].values #Variabilele de la mine din model
X = np.column_stack((Titanic["Pclass"].values, Titanic["Age"].values))

#Modelul de la b)

with pm.Model() as logistic_model:
    #coeficientii mei
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=2, shape=2)
    

    logits = alpha + pm.math.dot(X, beta)     # datele mele dupa ce fac exuatia a + bx+..
    

    observed = pm.Bernoulli('observed', pm.math.sigmoid(logits), observed=Y) #datele obs cu distributia beronoulli si folosind sigmoida
    
    # Eșantionare
    trace = pm.sample(1000, return_inferencedata=True, target_accept=0.95)


#Graficele :
az.plot_trace(trace, var_names=["alpha", "beta"])
plt.savefig('trace_plot.png')
az.plot_posterior(trace, var_names=["alpha", "beta"])
plt.savefig('posterior_plot.png')
plt.show()

#c
""" 
PClass are o influenta semnificativa deoarece in datele noastre,pasagerii din clasele superioare au rate mai mari de supravietuire.
""" 

#d

vec = np.array([[2, 30]])

#Scot samples-urile post esantionate
alpha_samples = trace.posterior['alpha'].values.flatten()
beta_samples = trace.posterior['beta'].values
# scot valorile pe modelul de mai sus
values = alpha_samples[:, None] + np.dot(vec, beta_samples.reshape((2, -1)))
#valorile dupa ce aplic sigmoid-ul
prob_suprav = 1 / (1 + np.exp(-values))
# media si intervalul 90% HDI
medieProb = np.mean(prob_suprav)
hdiProb = az.hdi(prob_suprav, hdi_prob=0.9)
#rezultate :
print(f"Probabilitate medie de supravietuire: {medieProb}")
print(f"Intervalul 90% HDI: {hdiProb}")

