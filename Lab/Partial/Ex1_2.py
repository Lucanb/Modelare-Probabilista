import random
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def arunca_moneda(sansa):
    return random.choices([0, 1], weights=[1 - sansa, sansa])[0]

#Dupa modelul din enunt am fct reteaua bayesiana
model = BayesianModel([('Jucator1', 'AruncaMoneda'),
                       ('AruncaMoneda', 'Jucator2'),
                       ('Jucator2', 'ObtineStema')])

#aici am parametrii conditionali like prob cond
data = {'Jucator1': [],
        'AruncaMoneda': [],
        'Jucator2': [],
        'ObtineStema': []}

for _ in range(2000):
    jucator1 = random.choice([0, 1])
    arunca_moneda_val = arunca_moneda(0.5)
    jucator2 = sum([arunca_moneda(1/3) for _ in range(arunca_moneda_val + 1)])
    obtine_stema = jucator2  #aici cand jucatorul 2 are stema

    data['Jucator1'].append(jucator1)
    data['AruncaMoneda'].append(arunca_moneda_val)
    data['Jucator2'].append(jucator2)
    data['ObtineStema'].append(obtine_stema)

df = pd.DataFrame(data)


model.fit(df, estimator=MaximumLikelihoodEstimator) #Aprox parametrilor in sens MLE

    
inference = VariableElimination(model)
result = inference.query(variables=['Jucator1', 'Jucator2'], evidence={'AruncaMoneda': 1}) #Estimarea inferenței

print(result)
