---
title: AI - 神经网络（Neural Network）
description: 神经网络（Neural Network）基本概念介绍。
keywords: AI,AI基础,神经网络,Neural Network
date: 2025-04-13 16:00:00 +0800
categories: [AI]
tags: [AI, AI Basic]
pin: true
math: true
mermaid: true
---

整体架构  

  - **线性方程**

  $$f(x) = Wx$$

  - **非线性方程**

  $$f=W_2 \max (0, W_1x) $$

  - *计算结果：*

  $$
  \begin{Bmatrix}
    x_1 \\
    x_2 \\
    x_3 \\
    ... \\
    x_{3072} \\
  \end{Bmatrix}
  \times
  \begin{Bmatrix}
    w1_{1,1} & w1_{1,2} & ... & w1_{1,100} \\
    w1_{2,1} & w1_{2,2} & ... & w1_{2,100} \\
    w1_{3,1} & w1_{3,2} & ... & w1_{3,100} \\
    \vdots & \vdots & \ddots & \vdots \\
    w1_{3072,1} & w1_{3072,2} & ... & w1_{3072,100} \\
  \end{Bmatrix}
  =
  \begin{Bmatrix}
    h_1 \\
    h_2 \\
    h_3 \\
    ... \\
    h_{100} \\
  \end{Bmatrix}
  $$


  $$
  \begin{Bmatrix}
    h_1 \\
    h_2 \\
    h_3 \\
    ... \\
    h_{100} \\
  \end{Bmatrix}
  \times
  \begin{Bmatrix}
    w2_{1,1} & w2_{1,2} & ... & w2_{1,10} \\
    w2_{2,1} & w2_{2,2} & ... & w2_{2,10} \\
    w2_{3,1} & w2_{3,2} & ... & w2_{3,10} \\
    \vdots & \vdots & \ddots & \vdots \\
    w2_{100,1} & w2_{100,2} & ... & w_{100,10} \\
  \end{Bmatrix}
  =
  \begin{Bmatrix}
    s_1 \\
    s_2 \\
    s_3 \\
    ... \\
    s_{10} \\
  \end{Bmatrix}
  $$

- **基本机构**

  $$f=W_2 \max (0, W_1x) $$

  *继续堆叠一层：*

$$f=W_3 \max (0, W_2 \max (0, W_1x)) $$

神经网络的强大之处在于，用更多的参数来拟合复杂的数据。


> 参数多到多少呢？百万级别都是小儿科了，但是参数越多越好吗？
{: .prompt-info }


- **参数初始化**

通常使用随机策略来参数初始化。

$$W = 0.01 * np.random.random(D, H) $$

## 1. 线性函数  

### 1.1 基本概念  

**线性回归**

基本概念，从`输入 --> 输出` 的映射 $ f(x, W) $ 。

1. $x$ : 输入向量，比如 image；

2. $W$ : 权重矩阵，parameters。

预测一个标签 $y$，它与 $x_1$、$x_2$ 都有关系。

> 那么它们是什么关系？
> 是 $x_1$ 对结果影响比较大，还是 $x_2$ 对结果影响比较大？
{: .prompt-info }


于是引入2个参数，一个$w_1$，一个$w_2$，结果如下：

$$
\begin{equation}
y = W_1x_1 + W_2x_2
\end{equation}
$$

这是一个基础的线性回归。$x_1$ 、 $x_2$ 都是已知的，$W_1$、$W_2$ 是未知的，需要通过训练数据来学习，求得$W_1$、$W_2$。

> $W_1$、$W_2$ 就是权重。
{: .prompt-tip }

### 1.2 数学表示  

$$
\begin{equation}
f(x,W) = Wx + b
\end{equation}
$$

神经网络中各层，$f(x,W)$ 是$10\times1$，$W$ 是$10\times3072$，$x$是$3072\times1$，$b$是$10\times1$.

> 权重参数、偏置、输出都是矩阵。
{: .prompt-tip }

假设一个 $32\times32\times3$ 的图像，$32\times32$是图片像素，也就是1024个点，$3$是R、G、B颜色通道，每个点都有3个值，也就是3072个值。

$32\times32\times3$的图像的矩阵表示：
$$
\begin{Bmatrix}
	x_1 \\
	x_2 \\
	x_3 \\
	... \\
	x_{3072} \\
\end{Bmatrix}
$$

每个点都有对应的权重参数表示：

$$
\begin{Bmatrix}
	W_1 & W_2 & W_3 & ... & W_{3072}
\end{Bmatrix}
$$

权重 $\times$ 输入，得到输出：

$$
y_1
$$

做10分类，输出10个值，需要10组权重参数矩阵。

$$
\begin{Bmatrix}
	W_{1,1} & W_{1,2}  & ... & W_{1,3072} \\
	W_{2,1} & W_{2,2}  & ... & W_{2,3072} \\
	\vdots & \vdots & \ddots & \vdots \\
	W_{10,1} & W_{10,2}  & ... & W_{10,3072} \\
\end{Bmatrix}
$$


在数学中用符号表示：

$$
A \in \mathbb{R}^{10 \times 3027}
$$

表示矩阵 $A$ 是一个维度为 $10 \times 3027$ 的实数矩阵。

**解释**

1. $\mathbb{R}$ 表示矩阵的元素属于实数集（所有元素都是实数）。
2. **上标** $10 \times 3027$：
   - $10$ 表示矩阵的行数（rows）。
   - $3027$ 表示矩阵的列数（columns）。
   - 所以，$A$ 是一个 包含 10 行、 3027 列的矩阵。

**示例：**

10组权重参数与输入相乘，得到10个输出：
$$
\begin{Bmatrix}
	y_1 \\
	y_2 \\
	y_3 \\
	... \\
	y_{10} \\
\end{Bmatrix}
$$

偏置是对结果做微调，给10个结果做微调，所以也是10个。

$$
\begin{Bmatrix}
	b_1 \\
	b_2 \\
	b_3 \\
	... \\
	b_{10} \\
\end{Bmatrix}
$$

最终表示为：

$$
f(x, W) =
\begin{Bmatrix}
	W_{1,1} & W_{1,2}  & ... & W_{1,3072} \\
	W_{2,1} & W_{2,2}  & ... & W_{2,3072} \\
	\vdots & \vdots & \ddots & \vdots \\
	W_{10,1} & W_{10,2}  & ... & W_{10,3072} \\
\end{Bmatrix}
\times
\begin{Bmatrix}
	x_1 \\
	x_2 \\
	x_3 \\
	... \\
	x_{10} \\
\end{Bmatrix}
+
\begin{Bmatrix}
	b_1 \\
	b_2 \\
	b_3 \\
	... \\
	b_{10} \\
\end{Bmatrix}
$$

也即是

$$
\begin{equation}
f(x,W) = Wx + b
\end{equation}
$$

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> 一切在计算机中都是矩阵，W和b都是随机初始化。
{: .prompt-tip }



### 1.3 计算方法  


> 一开始的权重，是随机初始化的。
{: .prompt-info }

$$
\begin{Bmatrix}
	0.2 & -0.5 & 0.1 & 2.0 \\
	1.5 & 1.3 & 2.1 & 0.0 \\
	0 & 0.25 & 0.2 & -0.3 \\
\end{Bmatrix}
\times
\begin{Bmatrix}
	56 \\
	231 \\
	24 \\
  2 \\
\end{Bmatrix}
+
\begin{Bmatrix}
	1.1 \\
	3.2 \\
	-1.2 \\
\end{Bmatrix}
=
\begin{Bmatrix}
	-96.8 \\
	437.9 \\
	63.95 \\
\end{Bmatrix}

$$

也即是

$$ 
W \times x_i + b = f(x_i, W, b)
$$

## 2. 损失函数  

**公式分解：**

$$
\begin{equation}
  L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_j}+1)
  \label{eq:series12}
\end{equation}
$$

**符号解释：**

1. $L_i$
   - 第 $i$ 个样本（或数据点）的**损失值**（Loss）。
   - 衡量该样本的分类错误程度，值越大表示分类越不准确。
2. $\sum_{j \neq y_i}$
   - 对所有**不等于正确类别** $y_i$ 的类别 $j$ 进行求和。
   - 即计算所有**错误类别**的损失贡献。
3. $\max(0, \cdot)$
   - **合页函数（Hinge Loss）**，表示只有当内部的值 > 0 时才计入损失。
   - 如果 $s_j - s_{y_i} + 1 \leq 0$，说明分类正确（无损失），否则损失为正。
4. $s_j$
   - 模型对类别 $j$ 的预测得分。
   - 在 SVM 中，$s_j = W_j^T x + b_j$。
5. $s_{y_i}$
   - 模型对**真实类别** $y_i$ 的预测得分。
   - 我们希望 $s_{y_i}$ 远大于 $s_j$ 。
6. +1 **（Margin）**
   - 一个**安全边界（margin）**，强制正确类别的得分至少比其他类别高 1.
   - 如果 $s_{y_i} \geq s_j + 1 $，则损失为 0；否则，损失线性增长。

**直观理解：**
- 这个损失函数的目标是让**正确类别的得分** $s_{y_i}$ 比所有错误类别得分 $s_j$ 高出至少1。
- 如果某个错误类别的得分 $s_j$ 比 $s_{y_i} - 1$ 还高，就会产生损失 $s_j - s_{y_i} + 1$。
- 最终损失 $L_i$ 是所有错误类别的损失之和。
​

**例子：**


假设：

- 类别：猫（$y_i=1$）、狗（$y_i=2$）、鸟（$y_i=3$）。
- 模型预测得分：$s=[3.2,5.1,2.0]$（即 $s_{y_i}=s_1=3.2$，$s_2=5.1$，$s_3=2.0$）。

$$
\begin{Bmatrix}
	cat \\
	dog \\
	bird \\
\end{Bmatrix}
=
\begin{Bmatrix}
	3.2 \\
	5.1 \\
	2.0 \\
\end{Bmatrix}
$$

计算损失：

$$
L_i = \max(0, 5.1 - 3.2 + 1) + \max(0, 2.0 - 3.2 + 1) 
    = \max(0, 2.9) + \max(0, -0.2)
    = 2.9 + 0
    = 2.9$$

- 狗（$j=2$）的得分比猫高 2.9，因此贡献了 2.9 的损失。
- 鸟（$j=3$）的得分低于猫，不贡献损失。

**用途：**
- 主要用于多类 SVM 和某些神经网络分类器（如使用 Hinge Loss 的模型）。

- 优化目标是让正确类别的得分显著高于其他类别，从而提升分类鲁棒性。

 
 **变体：**
如果去掉 $+1$ ，则公式变为：

$$L_i=\sum _{j \neq y_i} \max(0, s_j-s_{y_i})$$

此时仅要求正确类别的得分高于其他类别，但没有安全边界。

在二分类情况下，退化为标准的 SVM 合页损失：

$$  L_i=\max(0, 1 - y_j \cdot s) $$

其中 $ y_j \in \{-1, +1\} $。

---


> 如何损失函数的值相同，那么意味着两个模型一样吗？
{: .prompt-info }

$L=\frac{1}N \sum_{i=1}^N \max (0, f(x_i,W)_j - f(x_i, W)_{y_i} + 1 )$

输入数据：$ x = [1, 1, 1, 1]$
模型A：$W_1=[1, 0, 0, 0]$
模型B：$W_2=[0.25, 0.25, 0.25, 0.25]$

结果：$ W_1^Tx=W_2^Tx=1$

两组预测值都是1，完全相同，说明损失是一样的。

$W_1$ 这组数值，有的特别大，有的特别小。
$W_2$ 这组数值，都是0.25，相对均衡。

---
$$损失函数 = 数据损失 + 正则化惩罚项（\lambda R(W)）$$

$$L=\frac{1}N \sum_{i=1}^N \sum _{j \neq y_i} \max (0, f(x_i,W)_j - f(x_i, W)_{y_i} + 1 ) + \lambda R(W)$$

正则化惩罚项：$$R(W) = \sum_{k} \sum_{i} W_{k,l}^2$$
**W 的平方和再考虑正则。$\lambda$ 惩罚系数。**

## 3. Softmax 分类器

> 现在我们得到的是一个输入的得分值，但如果给我一个概率值岂不更好！
> 如何把一个得分值转换成一个概率值呢？
{: .prompt-info }

$$g(z)= \frac{1} {1+e^{-z}}$$

归一化：$P(Y = k \vert X=x_i) = \frac{e^sk} {\sum _i e^sj}$ where $ s = f(x_i, W) $

计算损失值：$L = - \log P(Y = y_i \vert X=x_i)$

$$
\begin{Bmatrix}
cat \\
car \\ 
frog \\
\end{Bmatrix}
=
\begin{Bmatrix}
3.2 \\
5.1 \\ 
-1.7 \\
\end{Bmatrix}
 {-exp ->}
\begin{Bmatrix}
24.5 \\
164.0 \\ 
0.18 \\
\end{Bmatrix}
 {-normalize ->}
\begin{Bmatrix}
0.13 \\
0.87 \\ 
0.00 \\
\end{Bmatrix}
 {->}
 L_i=-log(0.13)=0.89
$$


## 3. 前向传播  

得出损失值：$ f = Wx；L_i=\sum _{j \neq y_i} \max (0, s_j - s_{y_i} + 1 )$

### 3.1. 梯度下降

> 引入：当我们得到了一个目标函数后，如何进行求解？

直接求解？（并不一定可解，线性回归可以当作是一个特例）


> 常规套路：机器学习的套路就是我交给机器一堆数据，然后告诉它

什么样的学习方式是对的（目标函数），然后它朝着这个方向去做


> 如何优化：一口吃不成胖子，我们要静悄悄的一步步的完成迭代

（每次优化一点点，累计起来就是个大成绩了）


目标函数：$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})$$

寻找山谷的最低点，也就是我们的目标函数终点。
（什么样的参数能使得目标函数达到极值点）

下山分几步走呢？（更新参数）
（1）：找到当前最合适的方向
（2）：走那么一小步，走快了该“跌倒”了
（3）：按照方向与步伐去更新我们的参数

### 3.2. 目标函数

目标函数：$J(\theta) = \frac1{2m} \sum_{i=1}^m (y^i - h_\theta(x^i))^2$

- 批量梯度下降：
  $\frac{\partial J(\theta)}{\partial\theta _j} = -\frac1m \sum_{i=1}^m (y^i - h_\theta(x^i)) x_j^i$
  $\theta_j^{\prime} = \theta + \frac 1m  \sum_{i=1}^m (y^i - h_\theta(x^i)) x_j^i$
  （容易得到最优解，但是由于每次考虑所有样本，速度很慢）
- 随机梯度下降：
  $\theta_j^{\prime} = \theta_j +  (y^i - h_\theta(x^i)) x_j^i$
  （每次考虑一个样本，迭代速度很快，但是不一定每次都朝着收敛的方向）
- 小批量梯度下降：
  $\theta_j := \theta_j - \alpha \frac 1{10}  \sum_{k=i}^{i+9} (h_\theta(x^{(k)}) - y^{(k)})x_j^{k}$
  （每次考虑一小批样本，速度适中，收敛效果较好）

学习率（步长）：对结果会产生巨大的影响，一般小一些

如何选择：从小的时候，不行再小

批处理数量：32， 64， 128 都可以，很多时候还得考虑内存和效率

## 4. 反向传播

$f(x, y, z) = (x + y)x$

**e.g.**  
$x= -2, y= 5, z= -4$

$ q = x+y$  -> $\frac{\partial q}{\partial x} = 1, \frac{\partial q}{\partial y} = 1$

$ f = q+z$  -> $\frac{\partial f}{\partial q} = z, \frac{\partial f}{\partial z} = q$

Want: $\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z}$


## 5. 链式法则

梯度是一步一步传的

x -> f：
f($\frac{\partial z}{\partial x}$) => x：反向传播，$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial x}$

y -> f：
f($\frac{\partial z}{\partial y}$) => y：反向传播，$\frac{\partial L}{\partial y}=\frac{\partial L}{\partial z}\frac{\partial z}{\partial y}$

f -> z：
z => f：反向传播，$\frac{\partial L}{\partial z}$


复杂的例子：

$$f(w,x)=\frac 1 {1+e^{(-w_0x_0-w_1x_1-w_2)}}$$

[local gradient] x [its gradient]
$x0:[2] \times [0.2] = 0.4$
$w0: [-1] \times [0.2] = -0.2$

$f(x)=e^x -> \frac{df}{dx} = e^x$
$f(x)=ax -> \frac{df}{dx} = a$

$f(x)=\frac 1x -> \frac{df}{dx} = -\frac 1{x^2}$
$f(x)=c+x -> \frac{df}{dx} = 1$


可以一大块一大块的计算吗？

$f(w,x) = \frac 1 {1+e^{(w_0x_0+w_1x_1+w_2)}}$

sigmoid function：$\sigma(x)=\frac 1 {1+e^{-x}}$

$$
\begin{equation}
  \frac{d\sigma(x)}{dx} = \frac {e^{-x}} {(1+e^{-x})^2} = (\frac {1+e^{-x}-1} {(1+e^{-x})}) (\frac 1 {1+e^{-x}}) = (1- \sigma(x))\sigma(x)
\end{equation}
$$

## 6. 激活函数

### 6.1 sigmoid

$$
\begin{equation}
  \sigma(x) = \frac 1 {1+e^{-x}}
\end{equation}
$$


### 6.2 ReLU

$$
\begin{equation}
  \sigma(x) = \max(0,x)
\end{equation}
$$

