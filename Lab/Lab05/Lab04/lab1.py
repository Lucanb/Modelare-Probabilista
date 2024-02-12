import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
def Ex_1(alpha,lambdA,sigma,miu):
    #np.random.seed(2)
    totalTime= []
    for i in range(100):
        clientsCount = np.random.poisson(lambdA)
        payTime= np.random.normal(miu, sigma, clientsCount)
        orderTime = np.random.exponential(alpha, clientsCount)
        timeCount = payTime + orderTime
        totalTime.extend(timeCount)


    plt.hist(totalTime, bins=20, density=True, alpha=0.8, color="black")
    plt.xlabel("Order")
    plt.ylabel("Statistics Value")
    plt.title("Ex1")
    plt.show()
    return True
def Ex_2_3(alpha,lambdA,sigma,miu):

    N = 100
    alpha = 0.0
    while True:
     queueTime = []
     count = 0
     clientsCount = stats.poisson.rvs(lambdA, size=N)
     for index in clientsCount:
         payTime = stats.norm.rvs(miu, sigma, size = index)
         orderTime = stats.expon.rvs(alpha, size = index)
         timeCount = payTime + orderTime

         sw = True
         for timeIndex in timeCount:
                if timeIndex >= 15:
                  sw = False
                queueTime.append(timeIndex)

         if sw:
            count += 1

     if count/N <= 0.95:
        break

     mean = sum(queueTime) / len(queueTime)
     alpha += 0.01
    return alpha,mean

if __name__ == '__main__':

    alpha = 0.8
    lambdA = 20
    sigma = 0.5
    miu = 2
    #val =  Ex_1(alpha,lambdA,sigma,miu)
    ex_2Res,ex_3Res = Ex_2_3(alpha,lambdA,sigma,miu)
    print(ex_2Res)
    print(ex_3Res)