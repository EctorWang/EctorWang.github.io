---
layout: article
title: PINNs-Burgers-2D
tags:
  - PINNs
mathjax: true
---
> [!info] 文件创建时间
> 2025-05-07,16:42

## 二维 Burgers Equation 代码

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nt = 500
nx = 51
ny = 51

nu = 0.1
dt = 0.001

dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

comb = np.zeros((ny, nx))

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

un = np.zeros((ny, nx))
vn = np.zeros((ny, nx))

uf = np.zeros((nt, ny, nx))
vf = np.zeros((nt, ny, nx))

u = np.ones((ny, nx))
v = np.ones((ny, nx))

uf = np.ones((nt, ny, nx))
vf = np.ones((nt, ny, nx))

u[int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5
v[int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5

uf[0, int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5
vf[0, int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale")
plt.show()

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, v[:], cmap='jet')
plt.title("v solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("v scale")
plt.show()

for n in range(1, nt):
    un = u.copy()
    vn = v.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i, j] = (un[i, j] - (un[i, j] * dt / dx * (un[i, j] - un[i - 1, j])) - vn[i, j] * dt / dy * (un[i, j] - un[i, j - 1])) + (nu * dt / (dx**2)) * (un[i + 1, j] - 2 * un[i, j] + un[i - 1, j]) + (nu * dt / (dx**2)) * (un[i, j - 1] - 2 * un[i, j] + un[i, j + 1])
            v[i, j] = (vn[i, j] - (un[i, j] * dt / dx * (vn[i, j] - vn[i - 1, j])) - vn[i, j] * dt / dy * (vn[i, j] - vn[i, j - 1])) + (nu * dt / (dx**2)) * (vn[i + 1, j] - 2 * vn[i, j] + vn[i - 1, j]) + (nu * dt / (dx**2)) * (vn[i, j - 1] - 2 * vn[i, j] + vn[i, j + 1])
            uf[n, i, j] = u[i, j]
            vf[n, i, j] = v[i, j]

    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale")

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, v[:], cmap='jet')
plt.title("v solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
plt.show()

X, Y = np.meshgrid(x, y)

u = uf[499, :, :]

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale")
plt.show()
```

## 二维的Burgers Equation

> [!info] Burgers 方程
> 方程分为 X Y 两个方向的方程：计算的时候就要考虑两个PDE 方程的满足

Burgers' Equation for Horizontal Velocity (u):
水平速度$u$的伯格斯方程：
$$
 \frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}=\nu\left(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}\right) 
$$
垂直速度$v$的伯格斯方程：
Burgers' Equation for Vertical Velocity (v):
$$
 \frac{\partial v}{\partial t}+u\frac{\partial v}{\partial x}+v\frac{\partial v}{\partial y}=\nu\left(\frac{\partial^{2}v}{\partial x^{2}}+\frac{\partial^{2}v}{\partial y^{2}}\right) 
$$
**参数说明**：

- $u$：水平速度分量

- $v$：垂直速度分量

- $t$：时间

- $x$、$y$：空间坐标

- $\nu$ ：运动粘度，反映流体粘性和扩散性质 。





## Burgers Equation的简要介绍
二维空间中描述流体运动的 Burgers 方程，分别为水平速度 $u$ 和垂直速度 $v$ 的方程：

- **水平速度** $u$ **的 Burgers 方程**：
$$\frac{\partial u}{\partial t}+ u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}=\nu(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}})
$$

- **垂直速度** $v$ **的 Burgers 方程**：
$$\frac{\partial v}{\partial t}+ u\frac{\partial v}{\partial x}+v\frac{\partial v}{\partial y}=\nu(\frac{\partial^{2}v}{\partial x^{2}}+\frac{\partial^{2}v}{\partial y^{2}})
$$

### 公式中参数的物理含义

- $u$ **和** $v$

- $u$ 是流体在水平方向（通常设为 $x$ 方向）的速度分量 ；$v$ 是流体在垂直方向（通常设为 $y$ 方向）的速度分量。它们描述了流体微团在空间中不同方向上的运动快慢和方向。

- $t$：表示时间，用于描述流体运动状态随时间的变化。

- $x$ **和** $y$：是空间坐标，用于确定流体微团在二维平面内的位置 ，$x$ 对应水平方向坐标，$y$ 对应垂直方向坐标。

- $$\nu$$：运动粘度（kinematic viscosity），它反映了流体的粘性特性，即流体内部抵抗相对运动的能力 。运动粘度等于动力粘度 $\mu$除以流体密度 $\rho$，即 $\nu=\frac{\mu}{\rho}$ 。值越大，流体粘性越强，内摩擦力越大。

### 公式推导过程（以水平速度 $u$ 的方程为例，垂直速度 $v$ 方程推导类似 ）

1. **质量守恒（连续性方程）与动量守恒原理基础**

- 从流体力学基本原理出发，主要依据<font color="red">质量守恒和动量守恒定律</font>。质量守恒要求流入和流出控制体的质量流量平衡；动量守恒则是牛顿第二定律 $F = ma$ 在流体中的应用，即作用在流体微团上的合力等于微团动量的变化率。

2. **对流项（非线性项）**

- $u\frac{\partial u}{\partial x}$ **和** $$v\frac{\partial u}{\partial y}$$：这两项称为对流项。以 $u\frac{\partial u}{\partial x}$ 为例，$\frac{\partial u}{\partial x}$ 表示水平速度 $u$ 沿 $x$ 方向的变化率，而 $u$ 是流体在 $x$ 方向的速度。该项描述了由于流体自身的运动（对流运动）导致的速度 $u$ 在 $x$ 方向上的变化。也就是流体微团在运动过程中，由于其所处位置的速度分布不均匀，而引起的速度变化。$v\frac{\partial u}{\partial y}$同理，是由于 $y$ 方向的速度 $v$ 导致的 $u$ 沿 $y$ 方向的变化。这两项体现了流体运动的非线性特征。

3. **非定常项**

- $\frac{\partial u}{\partial t}$：表示水平速度 $u$ 随时间 $t$ 的变化率，即由于时间推移而导致的速度 $u$ 的变化，反映了流体运动的非定常特性，比如随时间变化的风场等情况。

4. **扩散项（粘性项）**

- $\nu(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}})$：这是扩散项，基于傅里叶定律（热传导类似概念）在流体粘性扩散中的应用。$\frac{\partial^{2}u}{\partial x^{2}}$ 和 $\frac{\partial^{2}u}{\partial y^{2}}$ 分别是 $u$ 沿 $x$ 和 $y$ 方向的二阶导数，反映了速度梯度的变化率。$\nu$ 是运动粘度，该项表示由于流体粘性作用，使得速度在空间上发生扩散，就像热量从高温区向低温区扩散一样，速度也会从高速度梯度区域向低速度梯度区域扩散，体现了粘性力对流体速度分布的影响 。

Burgers 方程在研究粘性流体的流动，如湍流、边界层流动等方面有重要应用，同时也是研究非线性偏微分方程性质和数值解法的典型模型方程。



### 粘度的测试

在 Burgers 方程的扩散项 $\nu(\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}})$ 和 $\nu(\frac{\partial^{2}v}{\partial x^{2}}+\frac{\partial^{2}v}{\partial y^{2}})$ 中，关键参数为运动粘度$\nu$，下面我将详细阐述其物理含义和计算方法。

### 运动粘度$\nu$的物理含义

运动粘度$\nu$是表征流体粘性特性的一个重要物理量，反映了流体内部抵抗相对运动的能力。从微观层面理解，流体的粘性源于分子间的内聚力和分子的热运动。当流体内部存在速度差异时，即不同流层之间有相对运动，粘性会使较快流层对较慢流层产生一个拉力，较慢流层对较快流层产生一个阻力，这种内摩擦力阻碍了流体层间的相对滑动，这就是粘性力的体现。

运动粘度$\nu$与动力粘度$\mu$密切相关，它是动力粘度$\mu$与流体密度$\rho$的比值，即$\nu = \frac{\mu}{\rho}$。相较于动力粘度，运动粘度更侧重于反映流体在惯性力和粘性力共同作用下的运动特性。在相同的动力粘度下，密度较小的流体，其运动粘度较大，意味着在相同的外力作用下，该流体速度分布的变化相对更慢，粘性对其运动的阻碍作用相对更显著。

### 运动粘度$\nu$的计算方法

#### 理论计算

- 对于理想气体，在一定条件下可基于气体分子运动论推导运动粘度。根据气体分子运动论，理想气体的动力粘度$\mu$与气体的分子量$M$、温度$T$、分子有效直径$d$等参数有关，其理论公式为$\mu=\frac{5}{16}\sqrt{\frac{\pi k_{B}T}{\pi d^{2}}}\sqrt{\frac{M}{N_{A}}}$，其中$k_{B}$为玻尔兹曼常量，$N_{A}$为阿伏伽德罗常数 。得到动力粘度$\mu$后，结合气体密度$\rho$，通过$\nu = \frac{\mu}{\rho}$计算运动粘度。不过，该理论公式是基于理想气体假设，实际气体和液体的计算更为复杂。

- 对于液体，目前尚无普适的理论计算公式，但存在一些基于经验和半经验的理论模型，如 Andrade 公式$\mu = A e^{\frac{B}{T}}$，其中$A$和$B$是与液体性质有关的常数，$T$为热力学温度，通过该公式得到动力粘度后再结合密度计算运动粘度，但这些公式的适用范围有限。

#### 实验测量

- **毛细管法**：让流体在一定压力差下流过已知半径和长度的毛细管，根据泊肃叶定律$Q=\frac{\pi R^{4}\Delta P}{8\mu L}$（其中$Q$为体积流量，$R$为毛细管半径，$\Delta P$为毛细管两端的压力差，$L$为毛细管长度），测量流量$Q$等参数可计算出动力粘度$\mu$，再结合密度算出运动粘度$\nu$。

- **旋转法**：将两个同心圆筒之间充满待测流体，内筒固定，外筒以一定角速度旋转，通过测量内筒受到的粘性力矩，结合相关公式计算动力粘度，进而得到运动粘度。该方法适用于测量不同剪切速率下的粘度，常用于非牛顿流体和高粘度流体的测量。

- **落球法**：在一个装有待测流体的垂直圆筒中，让一个光滑小球在重力作用下自由下落，根据斯托克斯定律$F = 6\pi\mu r v$（其中$F$为小球受到的粘性阻力，$r$为小球半径，$v$为小球下落的稳定速度），通过测量小球下落的速度等参数计算动力粘度，再结合密度计算运动粘度。该方法适用于测量<font color=red>低粘度流体</font>。
- 

如果还想了解 Burgers 方程在特定场景的应用案例，或是对其他参数的计算感兴趣，欢迎随时和我说。



## 代码解读



```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # matplotlib 基础之上的升级版

nt = 500   # 时间划分数量
# 二维网格划分
nx = 51
ny = 51

nu = 0.1   # 方程参数：流体粘度系数
dt = 0.001   # 时间划分 元单元

# 长度为 2 X 2 的区域网格划分的的两个方向上的单元长度
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

comb = np.zeros((ny, nx))

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

un = np.zeros((ny, nx))
vn = np.zeros((ny, nx))

uf = np.zeros((nt, ny, nx))
vf = np.zeros((nt, ny, nx))

u = np.ones((ny, nx))
v = np.ones((ny, nx))

uf = np.ones((nt, ny, nx))
vf = np.ones((nt, ny, nx))

u[int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5
v[int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5

uf[0, int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5
vf[0, int(0.75 / dy):int(1.25 / dy + 1), int(0.75 / dy):int(1.25 / dy + 1)] = 5

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale")
plt.show()

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, v[:], cmap='jet')
plt.title("v solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("v scale")
plt.show()

for n in range(1, nt):
    un = u.copy()
    vn = v.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i, j] = (un[i, j] - (un[i, j] * dt / dx * (un[i, j] - un[i - 1, j])) - vn[i, j] * dt / dy * (un[i, j] - un[i, j - 1])) + (nu * dt / (dx**2)) * (un[i + 1, j] - 2 * un[i, j] + un[i - 1, j]) + (nu * dt / (dx**2)) * (un[i, j - 1] - 2 * un[i, j] + un[i, j + 1])
            v[i, j] = (vn[i, j] - (un[i, j] * dt / dx * (vn[i, j] - vn[i - 1, j])) - vn[i, j] * dt / dy * (vn[i, j] - vn[i, j - 1])) + (nu * dt / (dx**2)) * (vn[i + 1, j] - 2 * vn[i, j] + vn[i - 1, j]) + (nu * dt / (dx**2)) * (vn[i, j - 1] - 2 * vn[i, j] + vn[i, j + 1])
            uf[n, i, j] = u[i, j]
            vf[n, i, j] = v[i, j]

    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1
    v[0, :] = 1
    v[-1, :] = 1

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale")

X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, v[:], cmap='jet')
plt.title("v solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
plt.show()

X, Y = np.meshgrid(x, y)

u = uf[499, :, :]

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u[:], cmap='jet')
plt.title("u solution")
plt.xlabel("X")
plt.ylabel("Y")
colorbar = plt.colorbar(contour)
colorbar.set_label("u scale")
plt.show()
```

