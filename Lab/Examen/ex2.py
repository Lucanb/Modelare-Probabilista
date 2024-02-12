import numpy as np
import matplotlib as plt
from scipy import stats

# in aceasta functie vom numara pentru cate puncte se respecta inegalitate(ele respecta o distributie) -- tip Monte Carlo
def MonteCarloAprox(N,k):
    counter=0

    for i in range(N):
        x=stats.geom.rvs(0.3)
        y=stats.geom.rvs(0.5)
        if x>y*y:
            counter+=1
    value=counter/N
    print(f"Iteratia {k},valoarea : {value}")
    return value
N=10000 #ITERATIILE
k=30 #pai aici am un nr de valori dupa care fac media si deviatia standard


MonteCarloAprox(N,0) # aici fac o iteratie initiala

vals=[]
for i in range(k):
    value=MonteCarloAprox(N,i)
    vals.append(value)

avg=np.mean(vals) #aici scoatem media
sd=np.std(vals) # aici scoate deviata standard
print(f"Media este : {avg}")
print(f"Deviatia standard este : : {sd}")

