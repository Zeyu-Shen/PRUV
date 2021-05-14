import numpy as np
import matplotlib.pyplot as plt

#return log pdf
#beta: temperature
def lpdf(x, beta):
    return - (1 - beta) * 500 * beta * x ** 2 * (1 - x) ** 2 - beta * x ** 2

#annealed importance sampling
#n: number of temperatures
#ite: number of iterations
#return: x, sample; w, log-weight
def annealed_importance_sampling(n, ite):
    beta = [1]
    w = []; x = []
    for _ in range(n):
        beta.append(1 / n)
    for _ in range(ite):
        curw = 1
        curx = np.random.normal(0, 1)
        for i in range(n):
            nxtx = np.random.normal(curx, 1)
            acc_prob = lpdf(nxtx, beta[i]) - lpdf(curx, beta[i])
            if np.log(np.random.rand()) < acc_prob:
                curx = nxtx
            curw += lpdf(curx, beta[i + 1])
            curw -= lpdf(curx, beta[i])
        x.append(curx)
        w.append(curw)
    return x, w

if __name__ == '__main__':
    ite = 100000; n = 10
    x, w = annealed_importance_sampling(n, ite)