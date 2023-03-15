# Python-Implementation-of-ARMORF-Function-from-MATLAB-Toolbox-BSMART
Python Implementation of ARMORF Function from MATLAB Toolbox BSMART

ARMORF can be used to compute autoregressive factors and to measure [Granger causality](https://en.wikipedia.org/wiki/Granger_causality).

Reference: 
>M. Morf, etal, Recursive Multichannel Maximum Entropy Spectral Estimation, IEEE trans. GeoSci. Elec., 1978, Vol.GE-16, No.2, pp85-94.
            
>S. Haykin, Nonlinear Methods of Spectral Analysis, 2nd Ed. Springer-Verlag, 1983, Chapter 2
            
>Jie Cui, Lei Xu, Steven L. Bressler, Mingzhou Ding, Hualou Liang, BSMART: a Matlab/C toolbox for analysis of multichannel neural time series, Neural Networks, 21:1094 - 1104, 2008.
## :unicorn:How to use
```
import numpy as np
from matplotlib import pyplot as plt


# Create a toy series
x = [1]   
for i in range(30):
    x.append(1.1 * x[-1])

# Compute the AR factor
Ax, Ex = armorf(np.asarray(x).reshape(1, -1),1,len(x),2)

# Fit a new series using AR factor
y = [1, 1.1]
factor_0, factor_1 = np.squeeze(Ax)
for i in range(30):
    y.append(factor_0 * y[-1] + factor_1 * y[-2])

plt.figure()
plt.clf()
plt.plot(np.asarray(x)-0.5, label='x')  # Adjust the intercept of x for better visualization
plt.plot(y, label='y')
plt.legend()
plt.show()
```
