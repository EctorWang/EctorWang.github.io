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


## 有限差分的思想
 偏微分方程采用的有限差分法，这也就是我们常说的数值解。数值解不是分析性的，也不是使用相关的数学定律得到的结果。因此关于本章的相关内容需要了解如何去解偏微分方程。

热传导方程是一个偏微分方程，它描述了物理系统的温度随时间的变化情况。该方程通常写作：
$$
\frac{\partial u}{\partial t} = k\frac{\partial^{2} u}{\partial x^{2}},
$$
带有两个热源的热棒，两端的热源加热，最后的温度上升。其中温度随时间的变化

### 热容参数
热容是一种物质的属性，用于量化其储存热能的能力。

| 物质  | 比热容 \(C_{p,s}\)（焦/克·摄氏度） | 摩尔热容 \(C_{p,m}\)（焦/摩尔·摄氏度） |
| --- | ------------------------ | -------------------------- |
| 空气  | 1.012                    | 29.19                      |
| 铝   | 0.89                     | 24.2                       |
| 氩   | 0.5203                   | 20.786                     |
| 铜   | 0.385                    | 24.47                      |
| 花岗岩 | 0.790                    | —                          |
| 石墨  | 0.710                    | 8.53                       |
| 氦   | 5.1932                   | 20.786                     |
| 铁   | 0.450                    | 25.09                      |
| 铅   | 0.129                    | 26.4                       |
| 锂   | 3.58                     | 24.8                       |
| 汞   | 0.14                     | 27.98                      |


数值求导是一种通过使用有限差分公式（如中心差分法 ）计算两个邻近点之间的斜率，来近似求解函数导数的方法：

$$
f^\prime(x) \approx \frac{f(x + h) - f(x - h)}{2h}
$$

其中，$f^\prime(x)$ 是点 $x$ 处的导数，$h$ 是一个小的区间，$f(x)$ 是 $x$ 处的函数值。


### 数值求导方法

数值求导方法是利用离散数据点来近似求解函数导数的技术。常见的方法包括：

#### 一阶导数方法：

- 向前差分：
$$
 f^\prime(x) \approx \frac{f(x + h) - f(x)}{h}  
$$
- 向后差分：
$$
 f^\prime(x) \approx \frac{f(x) - f(x - h)}{h}  
$$
- 中心差分：
$$
 f^\prime(x) \approx \frac{f(x + h) - f(x - h)}{2h}   
$$
#### 二阶导数方法：

- 三点中心差分：

$$
 f^{\prime\prime}(x) \approx \frac{\frac{f(x + h) - f(x)}{h}-\frac{f(x) - f(x - h)}{h}}{h}  
 $$
 
$$
 f^{\prime\prime}(x) \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}  
$$
- 五点模板：
$$
 f^{\prime\prime}(x) \approx \frac{f(x + 2h) - 4f(x + h) + 6f(x) - 4f(x - h) + f(x - 2h)}{h^2}  
$$

三点中心差分和五点中心差分的区别？各有什么优缺点？



### 方程离散化 Equation discretization

方程离散化是将连续的数学方程转化为适合数值分析的离散形式的过程。

这一过程在解决众多科学和工程问题中至关重要，包括求解偏微分方程、模拟物理系统以及进行数值模拟等。

主要公式如下：

- 偏微分方程：
$$
\frac{\partial u}{\partial t} = k\frac{\partial u^{2}}{\partial x^{2}}
$$

- 二阶导数近似公式：
$$
f^{\prime\prime}(x) \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^{2}}
$$

- 一阶导数近似公式：
$$
f^{\prime}(x) \approx \frac{f(x + h) - f(x)}{h}
$$

- 离散化后的方程：
$$
\frac{u(t + dt, x) - u(t, x)}{dt} = k\frac{u(t, x + dx) - 2u(t, x) + u(t, x - dx)}{dx^{2}}
$$
- 方便编程采用的方程-下面代码中的方程
$$
u(t+dt,x)=dt\cdot\left(\frac{k}{dx^2}\right)\left(u(t,x+dx\right)-2u(t,x)+u\left(t,x-dx)\right)-u(t,x)
$$



## 代码讲解

```python
import numpy
from matplotlib import pyplot

length = 10  # 杆的长度
k = 0.89     # 铝的热容参数
temp_left = 100     # 左端热源温度
temp_right = 200     # 右端热源温度
total_sim_time = 10     # 总体时间

dx = 0.1     # meshgrid 元单元
x_vector = numpy.linspace(0, length, int(length / dx))     # 一维杆的网格划分点

dt = 0.0001     # 拟合元单元时间
t_vector = numpy.linspace(0, total_sim_time, int(total_sim_time / dt))

u = numpy.zeros([len(t_vector), len(x_vector)])     # 所有数据的维度，每一行表示某个时候杆上各点的温度

# 热源边界条件确定
u[0, 0] = 100
u[:, 0] = temp_left
u[:, -1] = temp_right

# 绘制 u[100] 这个时刻的杆上各点的温度线图
pyplot.plot(x_vector, u[100])
pyplot.ylabel("temp")
pyplot.xlabel("x")
pyplot.show()

# 根据有限元的思想计算各点的数值
# 上面的介绍中有讲这一步具体怎么去做的
for t in range(1, len(t_vector) - 1):
    for x in range(1, len(x_vector) - 1):
        u[t + 1, x] = ((k * (dt / dx**2)) * (u[t, x + 1] - 2 * u[t, x] + u[t, x - 1])) + u[t, x]

# 绘制通过数值计算-有限差分的概念 计算之后的图形绘制
pyplot.plot(x_vector, u[90000])
pyplot.ylabel("temp")
pyplot.xlabel("x")
pyplot.show()

pyplot.plot(x_vector, u[99999])
pyplot.ylabel("temp")
pyplot.xlabel("x")
pyplot.show()
```