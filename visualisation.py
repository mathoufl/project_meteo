import numpy as np
import matplotlib.pyplot as plt
import random as rd

x0r = 0
x0nr =0

n = 27
t = 20
M = np.random.rand(n,n)

for i in range(n):
    S = sum(M[i])
    for j in range(n):
        M[i, j] = M[i, j]/S

M = np.transpose(M)

xr = [0 for k in range(t)]
xnr = [0 for k in range(t)]
y = [k for k in range(t)]

for time in range(t):
    xr[time] = x0r
    xnr[time] = x0nr

    x0nr = np.where(M[:, x0nr] == max(M[:, x0nr]))[0][0]
    x0r = np.random.choice(np.arange(0, n), p=M[:, x0nr])

plt.plot(y, xr)
plt.plot(y, xnr)

print(M)
plt.show()
