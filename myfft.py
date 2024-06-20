from matplotlib import pyplot as plt
import numpy as np
import cv2

def my_fft1d(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N <= 1:
        return x
    even = my_fft1d(x[0::2])
    odd = my_fft1d(x[1::2])
    # ret = np.zeros(N, dtype=complex)
    # for k in range(N // 2):
    #     W = np.exp(-2j * np.pi * k / N)
    #     ret[k] = even[k] + W * odd[k]
    #     ret[k + N // 2] = even[k] - W * odd[k]
    # return ret
    T = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate((even + T * odd, even - T * odd))

def my_fft2d(matrix):
    # 每一行进行FFT
    fft_rows = np.array([my_fft1d(row) for row in matrix])
    # 每一列进行FFT
    fft_cols = np.array([my_fft1d(col) for col in fft_rows.T]).T
    
    return fft_cols

def my_ifft2d(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    N = matrix.shape[0] * matrix.shape[1]
    # 先共轭
    conj_matrix = np.conjugate(matrix)
    # FFT
    fft_conj = my_fft2d(conj_matrix)
    # 结果再共轭，并除以长度
    ret = np.conjugate(fft_conj) / N
    return ret

def homomorphic_filter(img, d0=100, rl=0.3, rh=2.0, c=2):
    '''
    d0: 截止频率
    rl, rh: 滤波器幅度范围
    '''
    gray = img.copy()
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    # ln
    gray = np.log(1 + gray)
    # fft
    gray_fft = my_fft2d(gray)
    # gray_fftshift = np.fft.fftshift(gray_fft)
    # gaussian filter
    # dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = M ** 2 + N ** 2
    H = (rh - rl) * (1 - np.exp(-c * (D / (d0 ** 2)))) + rl
    H = np.fft.ifftshift(H)
    dst_fft = H * gray_fft
    
    # ifft
    dst_ifft = np.real(my_ifft2d(dst_fft))
    # exp
    dst = np.exp(dst_ifft) - 1
    # 归一化
    dst = 255 * (dst - dst.min()) / dst.max()
    
    return dst

if __name__ == "__main__":

    test_img = np.zeros((128, 128), dtype=np.uint8)
    test_img[28:98, 28:98] = 1
    # test_img[28:38, 28:38] = 1

    plt.subplot(221), plt.imshow(test_img, cmap='gray')
    plt.title("origin_bigger"), plt.axis("off")
    
    # x = np.arange(128).reshape(128, 1)
    # y = np.arange(128).reshape(1, 128)
    # M = (-1) ** (x + y)

    fft_img = my_fft2d(test_img )
    # fft_img = my_fft2d(test_img * M)
    # fft_img = np.fft.fft2(test_img)
    
    plt.subplot(222), plt.imshow(np.abs(fft_img), cmap='gray')
    plt.title("fft_img"), plt.axis("off")

    fft_img = np.fft.fftshift(fft_img)
    plt.subplot(223), plt.imshow(np.abs(fft_img), cmap='gray')
    plt.title("fft_img_shift"), plt.axis("off")

    fft_img_enhance = np.log(1 + np.abs(fft_img))
    plt.subplot(224), plt.imshow(fft_img_enhance, cmap='gray')
    plt.title("fft_img_enhance"), plt.axis("off")
    cv2.imshow("1", fft_img_enhance / fft_img_enhance.max())
    cv2.waitKey(0)

    # ifft_img = np.abs(my_ifft2d(fft_img))
    # # ifft_img = np.real(my_ifft2d(fft_img)) * M
    # plt.subplot(223), plt.imshow(ifft_img, cmap='gray')
    # plt.title("ifft_img"), plt.axis("off")

    plt.show()