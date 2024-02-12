from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx


model = BayesianNetwork([('C', 'I'), ('I', 'A'), ('C', 'A')])
cpdC = TabularCPD(variable='C', variable_card=2,
                          values=[[0.9995], [0.0005]])
cpdI = TabularCPD(variable='I', variable_card=2,
                          values=[[0.99, 0.97],
                                  [0.01, 0.03]],
                          evidence=['C'], evidence_card=[2])
cpdA = TabularCPD(variable='A', variable_card=2,
                        values=[[0.9999, 0.98, 0.05, 0.02],
                                [0.0001, 0.02, 0.95, 0.98]],
                        evidence=['C', 'I'], evidence_card=[2, 2])
model.add_cpds(cpdC, cpdI, cpdA)
assert model.check_model()


inference = VariableElimination(model)
ex2_res = inference.query(variables=['C'], evidence={'A': 1})  #aici ne intereseaza C(1)
print(ex2_res)
ex3_res = inference.query(variables=['I'], evidence={'A': 0})  #aici ne intereseaza I(0)
print(ex3_res)


paint = nx.circular_layout(model)
nx.draw(model, pos=paint, with_labels=True, node_size=3000, font_weight='bold', node_color='red')
plt.show()