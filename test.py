import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_square_points(size, num_points):
    offset = 32
    points = []
    per_side = num_points // 4
    for i in range(per_side):
        # 上边
        points.append((offset, offset + i * size // per_side))
        # 右边
        points.append((offset + i * size // per_side, offset + size - 1))
        # 下边
        points.append((offset + size - 1, offset + size - 1 - i * size // per_side))
        # 左边
        points.append((offset + size - 1 - i * size // per_side, offset ))
    print(len(points))
    return points

def create_binary_matrix(size, points):
    matrix = np.zeros((128, 128), dtype=int)
    for (x, y) in points:
        matrix[x, y] = 1
    return matrix

# 矩阵大小和点数
matrix_size = 64  # 例如 10x10 的矩阵
num_points = 64   # 点数

# 生成正方形上的点
points = generate_square_points(matrix_size, num_points)

# 生成二值矩阵
binary_matrix = create_binary_matrix(matrix_size, points)

cv2.imwrite("11.jpg", binary_matrix * 255)

# 显示二值矩阵
plt.imshow(binary_matrix, cmap='gray')
plt.title('64 Points on a Square')
plt.show()