import numpy as np
import matplotlib.pyplot as plt

#return log pdf
#beta: temperature
def lpdf(x, beta):
    return - 1e5 * beta * x ** 2 * (1 - x) ** 2

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

#parallel tempering
#swaps: the number of swaps
#n: the number of iterations to proceed with metropolis hastings in each swap
#num: the number of chains to run in parallel
def parallel_tempering(n, swaps, num):
    tot_trajectory = np.zeros(n * swaps)
    x = np.zeros(num)
    beta = [1]; q = 0.1
    for _ in range(num - 1):
        beta.append(beta[-1] * q)
    for _ in range(swaps):
        trajectory = np.zeros([num, n])
        for i in range(num):
            trajectory[i, : ] = metropolis_hastings(x[i], np.zeros(n), 0.01, beta[i])
        tot_trajectory[_ * n : (_ + 1) * n] = trajectory[0, : ]
        for i in range(num - 1):
            acc_betaswap = lpdf(trajectory[i, -1], beta[i + 1]) + lpdf(trajectory[i + 1, -1], beta[i])
            acc_betaswap -= lpdf(trajectory[i, -1], beta[i]) + lpdf(trajectory[i + 1, -1], beta[i + 1])
            if np.log(np.random.rand()) < acc_betaswap:
                x[i], x[i + 1] = trajectory[i + 1, -1], trajectory[i, -1]
            else:
                x[i], x[i + 1] = trajectory[i, -1], trajectory[i + 1, -1]
    return tot_trajectory

if __name__ == '__main__':
    n = 100; swap = 100
    traj = parallel_tempering(n, swap, 10)
    plt.plot(range(n * swap), traj)
    plt.show()