import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
import random

#p1 =1/2
#p2 =1/3

def arunca_moneda(sansa):
    return random.choices([0, 1], weights=[1 - sansa, sansa])[0]   #aici face un random choice ca sa simulam aruncarea

def joc():

    n = arunca_moneda(0.5)
    m = sum([arunca_moneda(1/3) for _ in range(n + 1)])         #aici acel joc,like folosesc moneda masluita pt n+1

    return n, m

def simulare(numar_runde):
    castiguri_jucator1 = 0                             #aici incepem simularea pe 20000
    castiguri_jucator2 = 0

    for _ in range(numar_runde):                        #pt toate rundele dam in acest mod cu banul
        n, m = joc()
        if n == m:
            castiguri_jucator1 += 1
        else:
            castiguri_jucator2 += 1

    procentaj_castig_jucator1 = (castiguri_jucator1 / numar_runde) * 100
    procentaj_castig_jucator2 = (castiguri_jucator2 / numar_runde) * 100

    print(f"Jucatorul 1 are un procentaj de castig de {procentaj_castig_jucator1}%")       #sansa jucat 1 castig
    print(f"Jucatorul 2 are un procentaj de castig de {procentaj_castig_jucator2}%")      #sansa jucat 2 castig

simulare(20000)           #simulez de 20 000 ori
