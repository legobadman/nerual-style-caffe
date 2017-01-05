import numpy as np
from scipy.optimize import leastsq

def func(x, p):
    # A*sin(2*pi*k*x + theta)
    A, k, theta = p
    return A * np.sin(2 * np.pi * k * x + theta)

def residuals(p, y, x):
    return y - func(x, p)

x = np.linspace(-2 * np.pi, 0, 100)
A, k, theta = 10, 0.34, np.pi/6
y0 = func(x, [A, k, theta])
y1 = y0 + 2 * np.random.rand(len(x))

p0 = [7, 0.2, 0]

plsq = leastsq(residuals, p0, args=(y1, x))

print u"real args:", [A, k, theta]
print u"fitted args:", plsq[0]

import pylab as pl
pl.plot(x, y0, label=u"real")
pl.plot(x, y1, label=u"noised data")
pl.plot(x, func(x, plsq[0]), label=u"fitted data")
pl.legend()
pl.show()
