import numpy as np
import cv2

def means_filter(input_image, filter_size):
    '''
    均值滤波器
    :param input_image: 输入图像
    :param filter_size: 滤波器大小
    :return: 输出图像

    '''
    input_image_cp = np.copy(input_image)  # 输入图像的副本
    filter_template = np.ones((filter_size, filter_size))  # 空间滤波器模板

    if filter_size/2 ==0:
        pad_num=filter_size/2
    else:
        pad_num = int((filter_size - 1) / 2)

    input_image_cp = np.pad(input_image_cp, (pad_num, pad_num), mode="constant", constant_values=0)  # 填充输入图像
    m, n = input_image_cp.shape  # 获取填充后的输入图像的大小

    output_image = np.copy(input_image_cp)  # 输出图像

    # 空间滤波
    for i in range(pad_num, m - pad_num):
        for j in range(pad_num, n - pad_num):
             output_image[i, j] = np.sum(filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1]) / (filter_size ** 2)

    output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪

    return output_image

test_pic = '/Users/xinming/Documents/tianchi/dev_data/0bc58747-9a2e-43e1-af1f-cf0a41f9f2ba.png'
img = cv2.imread(test_pic)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
