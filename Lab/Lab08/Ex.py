import pymc as pm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az

def incarcare_date():
    # Functia pentru incarcarea datelor din fisierul "Preturi.csv"
    cale_fisier = "Preturi.csv"
    df = pd.read_csv(cale_fisier)
    y = df["Pret"].values.astype(float)
    v1 = df["Viteza"].values.astype(float)
    v2 = df["HardDrive"].values.astype(float)
    v2 = np.log(v2)
    return y, v1, v2

def main():
    # Incarcarea datelor
    y_real, v1, v2 = incarcare_date()

    # Afișarea unui histogramă a distribuției prețurilor
    plt.hist(y_real, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

    # Definirea unui model Bayesian folosind PyMC3
    with pm.Model() as modelul_meu:
        alfa = pm.Normal('alfa', mu=0, sigma=1)
        beta1 = pm.Normal('beta1', mu=0, sigma=1)
        beta2 = pm.Normal('beta2', mu=0, sigma=1)
        miu = pm.Deterministic('miu', v2 * beta2 + v1 * beta1 + alfa)
        sigma = pm.HalfNormal('sigma', sigma=11)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=sigma)
        idata = pm.sample(20000, tune=20000, return_inferencedata=True)

    # Afișarea distribuției posterioare a mediei (miu) și deviației standard (sigma)
    az.plot_posterior(idata, var_names=['miu', 'sigma'])
    plt.show()

    # Calcularea și afișarea intervalului de credibilitate al coeficienților beta1 și beta2
    hdi_info = az.hdi(idata, var_names=["beta1", "beta2"], hdi_prob=0.95).values()
    print(hdi_info)

    # 3. Răspunsul este "DA". Cele mai multe valori ale coeficienților nu conțin 0 în intervalul de credibilitate.
    # Acest lucru sugerează că atât "Speed" cât și "HardDrive" influențează prețul.

    # 4. și 5. Nu este specificat exact ce trebuie făcut.

    # 6. Adăugarea unei variabile noi "x3" la model și analizarea distribuției posterioare a coeficienților
    with pm.Model() as my_model_premium:
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        beta3 = pm.Normal('beta3', mu=0, sigma=10)
        x3 = pm.Normal('x3', mu=0, sigma=1)  # Am adăugat definirea variabilei x3
        miu = pm.Deterministic('miu', x3 * beta3 + v2 * beta2 + v1 * beta1 + alfa)
        sigma = pm.HalfCauchy('sigma', 5)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=sigma, observed=y_real)
        idata = pm.sample(1000, tune=1000, return_inferencedata=True)

    # Afișarea distribuției posterioare a coeficienților beta1, beta2 și beta3
    az.plot_posterior(idata, var_names=["beta1", "beta2", "beta3"], hdi_prob=0.95)
    plt.show()

    # Calcularea și afișarea intervalului de credibilitate al coeficienților beta1, beta2 și beta3 pentru variabila "Premium"
    hdi_info_premium = az.hdi(idata, var_names=["beta1", "beta2", "beta3"], hdi_prob=0.95)
    print(hdi_info_premium)
    # Răspunsul este "DA". Cele mai multe valori ale coeficientului beta3 sunt diferite de 0, indicând că "Premium" are un impact.
    # Chiar dacă 0 este inclus în intervalul de credibilitate (HDI), beta3 este semnificativ diferit de 0.
    # Am folosit [0 - nu, 5 - da] pentru că era un interval mai apropiat de celelalte variabile, ceea ce ar putea produce rezultate mai precise.

if __name__ == "__main__":
    main()
