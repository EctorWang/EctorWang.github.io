---
layout: article
title: PINNs-1D-Heat-Equation
tags:
  - PINNs
mathjax: true
---
> [!info] 文件创建时间
> 2025-05-06,15:38

## 一维热传导方程代码

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


## 一维热传导方程的推导简要介绍
### 一、一维热传导方程介绍

一维热传导方程是描述热量在一维物体（如细长杆）中传递规律的数学方程 。通过它可了解物体内部温度分布，以及温度随时间和空间的变化情况。在工程和科学领域，如材料热性能分析、设备散热设计等方面有广泛应用 。其偏微分方程形式为：
$$
\frac{\partial u}{\partial t} = k\frac{\partial^{2}u}{\partial x^{2}}
$$
其中，$u = u(x, t)$表示温度，是关于空间坐标$x$和时间$t$的函数 ；$k$为热扩散率（也叫热扩散系数） 。热扩散率$k = \frac{\lambda}{\rho c_p}$ ，这里$\lambda$是导热系数（反映材料传导热量的能力，单位$W/(mÂ·K)$ ），$\rho$是材料密度（单位$kg/m^3$ ），$c_p$是比热容（单位质量物质温度升高$1K$所需热量，单位$J/(kgÂ·K)$ ） 。

### 二、方程推导过程

1. **基本假设**

- 考虑一根均匀的一维物体（如金属棒），其横截面积$A$恒定不变 。

- 物体的热传导系数$\lambda$已知且保持恒定 。

- 只考虑沿物体长度方向$x$方向）的热传导，忽略其他方向（如径向等）的热传导 。

2. **涉及的物理定律和概念**

- **能量守恒定律**：在一个封闭系统中，能量不会凭空产生或消失，只会从一种形式转化为另一种形式。在热传导情境下，流入某部分物体的热量减去流出的热量，等于该部分物体热存储（即温度变化引起的内能变化） 。

- **热流量（热流密度）**：单位时间内通过单位面积的热量，用$q$表示 ，根据傅里叶定律，$q = -\lambda\frac{\partial u}{\partial x}$ 。负号表示热量总是从高温区域流向低温区域，$rac{\partial u}{\partial x}$是温度关于位置 $x$ 的梯度 。
$
1. **推导步骤**

- 取物体中一个微小的长度元$\Delta x$，在微小时间间隔$\Delta t$内，分析该微元段的能量变化 。

- **流入与流出热量**：

- 从左侧（$x$处）流入微元段的热流量为$q(x,t)A\Delta t$ ，由傅里叶定律$q(x,t)=-\lambda\frac{\partial u(x,t)}{\partial x}$，所以流入热量$Q_{in}=-\lambda A\Delta t\frac{\partial u(x,t)}{\partial x}$ 。

- 从右侧（$x + \Delta x$处）流出微元段的热流量为$q(x + \Delta x,t)A\Delta t$ ，即$Q_{out}=-\lambda A\Delta t\frac{\partial u(x + \Delta x,t)}{\partial x}$ 。

- **微元段热存储变化**：微元段的质量$m=\rho A\Delta x$ ，根据比热容定义，温度变化$\Delta u$时，内能变化（热存储变化）$\Delta Q = m c_p\Delta u=\rho A\Delta x c_p\frac{\partial u}{\partial t}\Delta t$ 。

- **根据能量守恒定律建立等式**：

流入热量减去流出热量等于微元段热存储变化，即$Q_{in}-Q_{out}=\Delta Q$ 。

$$
-\lambda A\Delta t\frac{\partial u(x,t)}{\partial x}-\left(-\lambda A\Delta t\frac{\partial u(x + \Delta x,t)}{\partial x}\right)=\rho A\Delta x c_p\frac{\partial u}{\partial t}\Delta t)
$$

- **化简等式**：

两边同时除以$A\Delta x\Delta t$ ，得到

$$
\frac{\lambda}{\rho c_p}\frac{\frac{\partial u(x + \Delta x,t)}{\partial x}-\frac{\partial u(x,t)}{\partial x}}{\Delta x}=\frac{\partial u}{\partial t}
$$


当$\Delta x \to 0$时，根据导数定义，
$$
\frac{\frac{\partial u(x + \Delta x,t)}{\partial x}-\frac{\partial u(x,t)}{\partial x}}{\Delta x}=\frac{\partial^{2}u}{\partial x^{2}}
$$
，令$k = \frac{\lambda}{\rho c_p}$（热扩散率），就得到一维热传导方程：

$$\frac{\partial u}{\partial t} = k\frac{\partial^{2}u}{\partial x^{2}}$$

