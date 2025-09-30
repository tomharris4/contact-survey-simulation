# Script for fitting quadratic to age-related estimation bias data

import numpy as np
import matplotlib.pyplot as plt

def func(x, a, b, c, d):
    return a * np.exp(b * (x-d)) + c

# True ages (x) and estimated ages (y) from Voelke et al. and Norja et al.
x = np.array([0,12,13,14,15,16,17,18,25,47,75])
y = np.array([0,13.85,15.53,17.69,18.68,19.93,20.49,22.01,28.741,46.572,69.895])
weights = np.array([10000,1,1,1,1,1,1,1,1,2.25,2])
p2 = np.poly1d(np.polyfit(x=x, y=y, deg=2, w=weights))

context = 0.44
b_age = 1.5/0.39

max_x = 110

print('Max b_age: ', (-1 /(context * 2 * p2.coef[0] * max_x + context * (p2.coef[1] - 1))))

xp = np.linspace(0, max_x, max_x)

xp_2 = [h + context*b_age*(p2(h)-h) for h in xp]

# Plot age bias function
_ = plt.plot(x[1:], y[1:], '.',
             xp, xp, '--',
             xp, xp_2, '-')

plt.title('Age estimation bias')
plt.xlabel('Age of contact')
plt.ylabel('Average estimated age')

plt.legend(['Experimental','No bias','$-0.0035x^{2}+1.188x$'])
plt.savefig('../Figures/Supplementary Material/Age_estimation_bias.pdf')