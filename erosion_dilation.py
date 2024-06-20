import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ori = cv2.imread(".\\images\\test5-256x256.bmp")  # 读取图像
    kernel1 = np.ones((3, 3), np.uint8)  # 结构元
    
    plt.subplot(221), plt.imshow(ori, cmap='gray')
    plt.title("origin"), plt.axis("off")

    # erosion_1 = cv2.erode(ori, kernel1, iterations=1)
    # plt.subplot(222), plt.imshow(erosion_1, cmap='gray')
    # plt.title("erosion time=1"), plt.axis("off")   

    # erosion_10 = cv2.erode(ori, kernel1, iterations=10)
    # plt.subplot(223), plt.imshow(erosion_10, cmap='gray')
    # plt.title("erosion time=10"), plt.axis("off")

    # erosion_50 = cv2.erode(ori, kernel1, iterations=50)
    # plt.subplot(224), plt.imshow(erosion_50, cmap='gray')
    # plt.title("erosion time=50"), plt.axis("off") 

    dialate_1 = cv2.dilate(ori, kernel1, iterations=1)
    plt.subplot(222), plt.imshow(dialate_1, cmap='gray')
    plt.title("dialate time=1"), plt.axis("off")   

    dialate_10 = cv2.dilate(ori, kernel1, iterations=10)
    plt.subplot(223), plt.imshow(dialate_10, cmap='gray')
    plt.title("dialate time=10"), plt.axis("off")

    dialate_50 = cv2.dilate(ori, kernel1, iterations=50)
    plt.subplot(224), plt.imshow(dialate_50, cmap='gray')
    plt.title("dialate time=50"), plt.axis("off") 

    plt.show()

