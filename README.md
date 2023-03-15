# Python-Implementation-of-ARMORF-Function-from-MATLAB-Toolbox-BSMART
Python Implementation of ARMORF Function from MATLAB Toolbox BSMART
## How to use
```
x = [1]
for i in range(30):
    x.append(1.1 * x[-1])

Ax, Ex = armorf(np.asarray(x).reshape(1, -1),1,len(x),2)

y = [1, 1.1]
factor_0, factor_1 = np.squeeze(Ax)
for i in range(30):
    y.append(factor_0 * y[-1] + factor_1 * y[-2])

plt.figure()
plt.clf()
plt.plot(np.asarray(x)-0.5, label='x')
plt.plot(y, label='y')
plt.legend()
plt.show()
```
