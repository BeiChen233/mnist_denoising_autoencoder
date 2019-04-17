# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 17:52:49 2019

@author: Bei Chen
"""

from keras.layers import Input, Dense
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

from keras.datasets import mnist
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
 

# 导入数据函数
def load_data():
    """
    """
    # 导入数据
    (x_train, _), (x_test, _) = mnist.load_data()
    
    # 数据标准化(图片的像素值被归一化到 0~1 之间)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    
    # 打印查看数据集
    print("\n"+"---------------MNIST数据集的大小-----------------"+"\n")
    print("训练集数据大小: " + str(x_train.shape))
    print("测试集数据大小: " + str(x_test.shape))
    
    return x_train, x_test

# 添加高斯噪声函数
def add_noise(x_train,x_test,sigma):
    """
    """
    #noise_factor = 0.5
    x_train_noisy = x_train + np.random.normal(loc=0.0, scale=sigma/255, size=x_train.shape) 
    x_test_noisy = x_test + np.random.normal(loc=0.0, scale=sigma/255, size=x_test.shape) 
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    return x_train_noisy, x_test_noisy

# 模型训练函数
def train(x_train,x_test,x_train_noisy,x_test_noisy,num_epochs=10, batch_size=128):
    
    # 输入层
    input_img = Input(shape=(28, 28, 1))
    
    # 定义encoder
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # 定义decoder
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # 定义整个自编码器模型（确定模型的输入输出）
    autoencoder = Model(inputs=input_img, outputs=decoded)
    
    # 编译自编码器模型（+定义损失函数和优化器）
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    #autoencoder.compile(loss='mean_squared_error', optimizer='sgd')
    
    # 打开一个终端并启动TensorBoard，终端中输入 tensorboard --logdir=/autoencoder
    print("\n"+"--------------------训练开始---------------------"+"\n")
    # 传递训练数据和参数
    autoencoder.fit(x_train_noisy, x_train, 
                    epochs = num_epochs, batch_size = batch_size,
                    shuffle=True, validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder', 
                                           write_graph=True, histogram_freq=0)])
    print("\n"+"--------------------训练结束---------------------"+"\n")
    
    # 可视化CNN中间层，打印出你的每一层的大小细节
    autoencoder.summary()
    
    # 预测输出（测试集输入查看器去噪之后输出）
    decoded_imgs = autoencoder.predict(x_test_noisy)
    
    return decoded_imgs

# 查看测试图片的去噪效果函数
def view(x_test,x_test_noisy,encoded_imgs):
    """
    """
    n = 5  # how many digits we will display
    plt.figure(figsize=(15, 9))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
     
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# 计算一幅图像的信噪比函数
def cal_psnr(im1,im2):
    """
    """
    # assert pixel value range is 0-1
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(1 / mse)
    
    return psnr

# 计算全部图像的平均信噪比函数
def Cal_AveragePSNR(image_array1,image_array2):
    """
    """
    #image_array1 = np.reshape(image_array1, (len(image_array1), 784))
    #image_array2 = np.reshape(image_array2, (len(image_array2), 784))
    
    # 先计算图片的数量
    num_sample = image_array1.shape[0]
    
    # 计算多幅图像的信噪比值之和
    psnr_sum = 0
    for i in range(num_sample):
        psnr = cal_psnr(image_array1[i],image_array2[i])
        psnr_sum = psnr_sum + psnr
    # 计算多幅图像的平均信噪比
    average_psnr = psnr_sum / num_sample
    
    return average_psnr


# main
if __name__ == '__main__':
    
    # 导入原始数据
    x_train, x_test = load_data()
    
    # 添加高斯噪声
    x_train_noisy, x_test_noisy = add_noise(x_train, x_test, sigma=50)
    
    # 经过训练得到的测试集重构输出
    decoded_imgs = train(x_train,x_test, x_train_noisy,x_test_noisy, 
                         num_epochs=10, batch_size=128)
    
    # 查看去噪效果
    view(x_test,x_test_noisy,decoded_imgs)
    
    # 计算Test集去噪图像的信噪比
    Average_PSNR = Cal_AveragePSNR(x_test,decoded_imgs)
    print("\n Test Average PSNR : %.2fdB " % Average_PSNR)










































