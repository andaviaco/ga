import numpy as np
import matplotlib.pyplot as plt, mpld3

import lib
from GAManager import GAManager


def evalFn(x):
    return x**4 + 5*x**3 + 4*x**2 - 4*x + 1

def evalFn2(x):
    return (x - 2) ** 2

def plot(fn, lim_low, lim_high, *, color='b', steps=0.01):
    ax = plt.subplot(111)

    t = np.arange(lim_low, lim_high, steps)
    s = fn(t)

    line, = plt.plot(t, s, color=color, lw=2)

    plt.show()

def main():
    # plot(evalFn, -4, -1)
    # plot(evalFn2, 0, 4, color='r')

    gam_sphere = GAManager(100, 100, lib.sphere, mutation_p=0.5, fn_lb=[-5, -5], fn_ub=[5, 5])
    gam_ackley = GAManager(100, 100, lib.ackley, mutation_p=0.5, fn_lb=[-20, -20], fn_ub=[20, 20])
    gam_rastrigin = GAManager(100, 100, lib.rastrigin, mutation_p=0.5, fn_lb=[-5, -5], fn_ub=[5, 5])
    # gam2 = GAManager(100, 100, evalFn2, evalsegments=21, llow=0, lhigh=4)

    print('RESULTS')

    min_val = gam_sphere.optimize()
    print('Sphere')
    print('MIN ≃', min_val)

    min_val = gam_ackley.optimize()
    print('Ackley')
    print('MIN ≃', min_val)

    min_val = gam_rastrigin.optimize()
    print('Rastrigin')
    print('MIN ≃', min_val)

if __name__ == '__main__':
    main()
