import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta

#exercitiul 1
grid_points = 1000
grid = np.linspace(0, 1, grid_points)
prior_simple = (grid <= 0.5).astype(int)
prior_abs = abs(grid - 0.5)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(grid, prior_simple, label='Prior Simple')
plt.title('Prior Simplu')
plt.subplot(1, 2, 2)
plt.plot(grid, prior_abs, label='Prior Abs')
plt.title('Prior Absolut')
plt.show()

#exercitiul 2
N_values = [100, 1000, 10000]
results = []

for N in N_values:
    inside_circle = 0
    for _ in range(N):
        x, y = np.random.rand(2)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    pi_estimate = 4 * inside_circle / N
    results.append(pi_estimate)

print("Estimări ale lui π pentru diferite valori N:", results)

#exercitiul 3
def metropolis(func, steps=1000):
    samples = np.zeros(steps)
    old_x = 0.5
    for i in range(steps):
        new_x = np.random.normal(old_x, 0.1)
        if new_x > 0 and new_x < 1:
            acceptance = func(new_x) / func(old_x)
            if acceptance >= np.random.rand():
                samples[i] = new_x
                old_x = new_x
            else:
                samples[i] = old_x
        else:
            samples[i] = old_x
    return samples

def beta_binomial(x, a=2, b=2, N=10, k=5):
    return beta.pdf(x, a, b) * binom.pmf(k, N, x)

samples = metropolis(beta_binomial)
plt.hist(samples, bins=50, density=True)
plt.title('Distribuția Beta-Binomială')
plt.show()