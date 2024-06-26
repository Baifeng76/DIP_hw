# 数字图像处理大作业

文件说明：

* 文件夹`images`：测试用图片
* `demo_v2.py`：主程序代码文件，运行指令`python demo_v2.py`。
* `my_fft.py`：FFT相关函数代码（主程序调用）
* `erosion_dilation.py`：大作业(6)的腐蚀、膨胀实验代码

## 1 大作业(1)

### 1.1 BMP&JPG文件读取、显示

打开图像，File>Open

转换到BMP文件：File>Convert to BMP

<img src=".\pics\大作业1-1.png" alt="大作业1-1" style="zoom:50%;" />

### 1.2 图像基本操作

#### 1.2.1 求反

Edit>Invert

<img src=".\pics\大作业1-2.png" alt="大作业1-2" style="zoom:50%;" />

#### 1.2.2 镜像

Edit>Mirror

<img src=".\pics\大作业1-3.png" alt="大作业1-3" style="zoom:50%;" />

#### 1.2.3 平移

Edit>Translate

<center class = "half">
    <img src=".\pics\大作业1-4-1.png" alt="大作业1-4-1" style="zoom:33%;" /><img src=".\pics\大作业1-4-2.png" alt="大作业1-4-2" style="zoom:33%;" />
</center>
#### 1.2.4 旋转

Edit>Rotate

<center class = "half">
    <img src=".\pics\大作业1-5-1.png" alt="大作业1-5-1" style="zoom:33%;" /><img src=".\pics\大作业1-5-2.png" alt="大作业1-5-2" style="zoom:33%;" />
</center>
#### 1.2.5 加

Edit>Add

<center class = "half">
    <img src=".\pics\大作业1-6.png" alt="大作业1-6" style="zoom:33%;" />
</center>

## 2 大作业(2)

傅里叶变换：Edit>FFT

傅里叶反变换：Edit>IFFT【若已经使用过FFT，会对FFT结果进行IFFT；否则，对打开的图像直接IFFT。】

### 2.1 观察典型图像FFT变换后的频谱

#### 2.1.1 测试图像FFT结果

<img src=".\pics\大作业2-1.png" alt="大作业2-1" style="zoom:50%;" />

#### 2.1.2 平移白色方块

<img src=".\pics\大作业2-2.png" alt="大作业2-2" style="zoom:50%;" />

#### 2.1.3 放大白色方块

<img src=".\pics\大作业2-3.png" alt="大作业2-3" style="zoom:50%;" />

#### 2.1.4 缩小白色方块

<img src=".\pics\大作业2-4.png" alt="大作业2-4" style="zoom:50%;" />

### 2.2 测试一幅自然图像FFT变换、FFT反变换

<img src=".\pics\大作业2-7.png" alt="大作业2-7" style="zoom:50%;" />

### 2.3 解释说明

> 右侧的图像是用下方方法得到的，解释原因。
>  (a)用(-1)x+y乘以左侧图像；
>  (b)计算DFT;
>  (c)取变换的复共轭
>  (d)计算傅里叶反变换
>  (e)用(-1)x+y乘以结果的实部
>
> <img src=".\pics\大作业2-6.png" alt="大作业2-6" style="zoom:33%;" />

这里主要利用到了二维傅里叶变换的平移性、共轭对称性和旋转性。

* 根据二维傅里叶变换的平移性，有$f(x,y)\cdot (-1)^{x+y}\Leftrightarrow F(u-N/2, v-M/2)$，因此步骤(a)和步骤(e)相当于对图像的频谱中心化或反中心化。
* 根据二维傅里叶变换的共轭对称性，有$F^*(u, v)=F(-u, -v)$，因此步骤(c)相当于对频谱做中心对称。
* 根据二维傅里叶变换的旋转性，变换旋转一个角度$\alpha$，其对应的图像也旋转相同的角度$\alpha$。

因此，每个步骤对频谱、图像的操作变化解释如下：

| 步骤                            | 对图像的等价操作   | 解释                                                         |
| ------------------------------- | ------------------ | ------------------------------------------------------------ |
| (a)用$(-1)^{x+y}$乘以左侧图像   | 图像频谱中心化     | 2D-DFT的平移性                                               |
| (b)计算DFT                      | 图像变换为频谱     | 注意频谱已中心化                                             |
| (c)取变换的复共轭               | 频谱和图像旋转180° | 2D-DFT的共轭对称性和旋转性，中心对称相当于将频谱旋转180°，图像同时也旋转180° |
| (d)计算傅里叶反变换             | 频谱转换为图像     | 此时图像是复数矩阵，且频谱是中心化的                         |
| (e)用$(-1)^{x+y}$乘以结果的实部 | 图像频谱反中心化   | 2D-DFT的平移性，恢复到图像，此时图像已被旋转180°             |

## 3 大作业(3)

### 3.1 直方图均衡化

Enhance>Hist Equalize

可以看出，第二次直方图均衡化结果与第一次结果相同。

<img src=".\pics\大作业3-1.png" alt="大作业3-1" style="zoom:50%;" />

### 3.2 同态滤波

Enhance>Homomorphic Filter

<img src=".\pics\大作业3-2.png" alt="大作业3-2" style="zoom:80%;" />

### 3.3 指数变换增强

Enhance>Exp Transform

#### 3.3.1 $\gamma < 1$的情况

<center class = "half">
    <img src=".\pics\大作业3-3-1.png" alt="大作业3-3-1" style="zoom:50%;" /><img src=".\pics\大作业3-3-2.png" alt="大作业3-3-1" style="zoom:50%;" />
</center>
#### 3.3.2 $\gamma > 1$的情况

<center class = "half">
    <img src=".\pics\大作业3-3-3.png" alt="大作业3-3-3" style="zoom:50%;" /><img src=".\pics\大作业3-3-4.png" alt="大作业3-3-4" style="zoom:50%;" />
</center>
### 3.4 Laplace锐化

Enhance>Laplace Sharpen

<img src=".\pics\大作业3-4.png" alt="大作业3-4" style="zoom:50%;" />

## 4 大作业(4)

### 4.1 图像边缘提取

Edge>Roberts或Sobel或Prewitt或Laplace

<img src=".\pics\大作业4.png" alt="大作业4" style="zoom:70%;" />

### 4.2 不同算子比较

* Roberts算子对陡峭的低噪声图像效果比较好，尤其是边缘正负45°较多的图像，计算量小，但定位准确率较差；
* Sobel算子能很好地消除噪声的影响，边缘提取效果较好，但计算量较大；
* Prewitt算子对灰度渐变的图像边缘提取效果比较好，但没有考虑相邻点的远近对当前像素点的影响；
* Laplacian算子对噪声比较敏感，其提取边缘的效果容易混入噪声的边缘。

## 5 大作业(5)

傅里叶描述子：Descriptor>Fourier Descriptor

下图分别是：原图（N=64）、M=2、M=16、M=62、M=64的图像。

<img src=".\pics\大作业5.png" alt="大作业5" style="zoom:100%;" />

## 6 大作业(6)

反复膨胀图像中的一组前景像素的极限效果是：图像最终变为全白（全1）。

比如，使用结构元为3x3的全1矩阵，膨胀次数为1、10、50时，测试效果如下。

<img src=".\pics\大作业6-1.png" alt="大作业6-1" style="zoom:50%;" />

反复腐蚀图像中的一组前景像素的极限效果是：图像最终变为全黑（全0）。

比如，使用结构元为3x3的全1矩阵，腐蚀次数为1、10、50时，测试效果如下。

<img src=".\pics\大作业6-2.png" alt="大作业6-2" style="zoom:50%;" />