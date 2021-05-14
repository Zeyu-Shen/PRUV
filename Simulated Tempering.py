import numpy as np
import matplotlib.pyplot as plt

#return log pdf
#beta: temperature
def lpdf(x, beta):
    return - 500 * beta * x ** 2 * (1 - x) ** 2

#proceeds with a metropolis step
def metropolis_hastings(x0, xt, h, beta):
    xt[0] = x0
    for _ in range(1, len(xt)):
        xnew = xt[_ - 1] + np.sqrt(2 * h) * np.random.randn()
        acc = lpdf(xnew, beta) - lpdf(xt[_ - 1], beta)
        if np.log(np.random.rand()) < acc:
            xt[_] = xnew
        else:
            xt[_] = xt[_ - 1]
    return xt

#simulated tempering
#ite: the total number of iterations
#n: the number of steps for metropolis hastings in each iteration
#num: the number of temperatures
def simulated_temperings(n, ite, num):
    trajectory = []
    beta = [1]; q = 0.1
    for _ in range(num - 1):
        beta.append(beta[-1] * q)
    x = 0; ind = 0
    Z = np.ones(num)
    for _ in range(ite):
        xnew = metropolis_hastings(x, np.zeros(n), 0.01, beta[ind])
        x = xnew[-1]
        if ind == 0:
            for i in range(len(xnew)):
                trajectory.append(xnew[i])
        if np.random.rand() < 0.5:
            nind = ind + 1
        else:
            nind = ind - 1
        if nind < 0 or nind >= num:
            continue
        acc_betamove = np.log(Z[ind]) + lpdf(x, beta[nind]) - np.log(Z[nind]) - lpdf(x, beta[ind])
        mul = 0
        for i in range(n):
            mul += np.exp((beta[ind] - beta[nind]) * lpdf(xnew[i], 1))
        mul /= n
        Z[nind] = Z[ind] * mul
        if np.log(np.random.rand()) < acc_betamove:
            ind = nind
    return trajectory

if __name__ == '__main__':
    n = 100; ite = 10000; num = 2
    traj = simulated_temperings(n, ite, num)
    plt.plot(range(len(traj)), traj)
    plt.show()