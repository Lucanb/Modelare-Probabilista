import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# cei 20o timpi medii asteptare
np.random.seed(42)
miu = 10
sigma = 2
timpimedii_asteptare = np.random.normal(miu, sigma, 200)


plt.hist(timpimedii_asteptare, bins=30, density=True, alpha=0.7, color='b') # vizualizare cu histogr
plt.title('Histograma Timpi Medii de Așteptare')
plt.xlabel('Timp Mediu de Așteptare')
plt.ylabel('Frecvență relativă')
plt.show()
