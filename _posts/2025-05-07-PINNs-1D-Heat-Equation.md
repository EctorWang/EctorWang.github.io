---
layout: article
title: PINNs-1D-Heat-Equation
tags:
  - PINNs
mathjax: true
---
> [!info] 文件创建时间
> 2025-05-07,15:38

## 一维热传导方程

```python
import numpy
from matplotlib import pyplot

length = 10
k = 0.89
temp_left = 100
temp_right = 200
total_sim_time = 10

dx = 0.1
x_vector = numpy.linspace(0, length, int(length / dx))

dt = 0.0001
t_vector = numpy.linspace(0, total_sim_time, int(total_sim_time / dt))

u = numpy.zeros([len(t_vector), len(x_vector)])

u[0, 0] = 100
u[:, 0] = temp_left
u[:, -1] = temp_right

pyplot.plot(x_vector, u[100])
pyplot.ylabel("temp")
pyplot.xlabel("x")
pyplot.show()

for t in range(1, len(t_vector) - 1):
    for x in range(1, len(x_vector) - 1):
        u[t + 1, x] = ((k * (dt / dx**2)) * (u[t, x + 1] - 2 * u[t, x] + u[t, x - 1])) + u[t, x]

pyplot.plot(x_vector, u[90000])
pyplot.ylabel("temp")
pyplot.xlabel("x")
pyplot.show()

pyplot.plot(x_vector, u[99999])
pyplot.ylabel("temp")
pyplot.xlabel("x")
pyplot.show()
```