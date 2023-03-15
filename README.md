# Python-Implementation-of-ARMORF-Function-from-MATLAB-Toolbox-BSMART
Python Implementation of ARMORF Function from MATLAB Toolbox BSMART
## How to use
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
