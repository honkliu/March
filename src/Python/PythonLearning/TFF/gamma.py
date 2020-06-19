import numpy as np
from scipy.special import gamma
from scipy.stats import gamma as ga
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,8))
# The Gamma function
x = np.linspace(-5, 5, 1000)
plt.plot(x, gamma(x), marker='+', ls='', c='k', label='$\Gamma(x)$')

# (x-1)! for x = 1, 2, ..., 6
x2 = np.linspace(1,6,6)
y = np.array([1, 1, 2, 6, 24, 120])
plt.plot(x2, y, marker='*', markersize=12, markeredgecolor='r',
           markerfacecolor='r', ls='',c='r', label='$(x-1)!$')

z = np.linspace(0, 15, 1000)
plt.plot(z, np.log(gamma(z)), ls='dashdot', 
            markerfacecolor='r', c='b', label='$log\Gamma(x)$')

plt.title('Gamma Function')
plt.ylim(-15,30)
plt.xlim(-5, 15)
plt.xlabel('$x$')
plt.legend()
plt.show()


alpha_values = [1, 2, 3, 3, 3]
beta_values = [0.5, 0.5, 0.5, 1, 2]
color = ['b','r','g','y','m']
x = np.linspace(1E-6, 10, 1000)

fig, ax = plt.subplots(figsize=(12, 8))

for k, t, c in zip(alpha_values, beta_values, color):
    dist = ga(k, loc=0, scale=1/t)
    plt.plot(x, dist.pdf(x), c=c, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (k, t))

plt.xlim(0, 10)
plt.ylim(0, 2)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|\alpha,\beta)$')
plt.title('Gamma Distribution')

plt.legend(loc=0)
plt.show()