import numpy as np
import matplotlib.pyplot as plt, mpld3

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
    plot(evalFn, -4, -1)
    # plot(evalFn2, 0, 4, color='r')

    gam = GAManager(100, 100, evalFn, mutation_p=0.01, evalsegments=16, llow=-4, lhigh=-1)
    # gam2 = GAManager(100, 100, evalFn2, evalsegments=21, llow=0, lhigh=4)

    optimal1 = gam.optimize()
    # optimal2 = gam2.optimize()

    print('RESULTS')
    print('x^4 + 5*x^3 + 4*x^2 - 4*x + 1')
    print('MIN: X≃', optimal1)
    # print('(x - 2)^2')
    # print('MIN: X≃', optimal2)


if __name__ == '__main__':
    main()
