# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:35:31 2019

@author: Bei Chen
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
 

# 导入数据函数
def load_data():
    """
    """
    # 导入数据
    (x_train, _), (x_test, _) = mnist.load_data()
    
    # 数据标准化
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    
    # 打印查看数据集
    print("\n"+"---------------MNIST数据集的大小-----------------"+"\n")
    print("训练集数据大小: " + str(x_train.shape))
    print("测试集数据大小: " + str(x_test.shape))
    
    return x_train,x_test

# 添加高斯噪声函数
def add_noise(x_train,x_test,sigma):
    """
    """
    #noise_factor = 0.5
    x_train_noisy = x_train + np.random.normal(loc=0.0, scale=sigma/255, size=x_train.shape) 
    x_test_noisy = x_test + np.random.normal(loc=0.0, scale=sigma/255, size=x_test.shape) 
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    return x_train_noisy,x_test_noisy

# 模型训练函数
def train(x_train,x_test,x_train_noisy,x_test_noisy,
          num_epochs=50, batch_size=128, hidden_num=200):
    
    # 定义隐藏层神经元个数
    encoding_dim = hidden_num
    # 输入层
    input_img = Input(shape=(784,))
    # 编码层
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # 解码层
    decoded = Dense(784, activation='sigmoid')(encoded)
    # 定义整个自编码器模型（确定模型的输入输出）
    autoencoder = Model(inputs=input_img, outputs=decoded)
    
    
    # 编码器模型
    encoder = Model(inputs=input_img, outputs=encoded)
    
    # 解码器模型
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
    
    # 编译自编码器模型（+定义损失函数和优化器）
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    #autoencoder.compile(loss='mean_squared_error', optimizer='sgd')
    
    print("\n"+"--------------------训练开始---------------------"+"\n")
    # 传递训练数据和参数
    autoencoder.fit(x_train_noisy, x_train, 
                    epochs=num_epochs, batch_size=batch_size, 
                    shuffle=True, validation_data=(x_test_noisy, x_test))
    print("\n"+"--------------------训练结束---------------------"+"\n")
    
    # 可视化自编码器中间层，打印出你的每一层的大小细节
    autoencoder.summary()
    
    encoded_imgs = encoder.predict(x_test_noisy)
    decoded_imgs = decoder.predict(encoded_imgs)
    
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

# 计算多幅图像的平均信噪比函数
def Cal_AveragePSNR(image_array1,image_array2):
    """
    """
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
                         num_epochs=50, batch_size=512, hidden_num=196)
    
    # 查看去噪效果
    view(x_test,x_test_noisy,decoded_imgs)
    
    # 计算Test集去噪图像的信噪比
    Average_PSNR = Cal_AveragePSNR(x_test,decoded_imgs)
    print(" Test Average PSNR : %.2fdB " % Average_PSNR)






























