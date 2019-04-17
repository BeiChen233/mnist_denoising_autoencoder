# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:05:00 2019

@author: Bei Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
import time
 
#from tf_utils import *

# 准备工作
ops.reset_default_graph()          #能够重新运行模型而不覆盖tf变量
print("TensorFlow version: %s" %tf.__version__)

# 导入MNIST类
mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False, seed=1)

# 制作训练、测试数据函数
def classify_data(mnist):
    """
    """
    ## 查看MNIST数据集
    X_train = mnist.train.images
    X_test = mnist.test.images
    print("\n"+"---------------MNIST数据集的大小-----------------"+"\n")
    print("训练集数据大小: " + str(X_train.shape))
    print("测试集数据大小: " + str(X_test.shape))
    
    return X_train, X_test

# 创建占位符函数
def create_placeholders(n_x):
    """
    """
    with tf.name_scope('inputs'):
        X = tf.placeholder(dtype=tf.float32,shape=[None,n_x],name='Noise_image')
        Y = tf.placeholder(dtype=tf.float32,shape=[None,n_x],name='Clear_image')
    
    return X, Y

# 初始化参数函数    
def initialize_parameters():
    """
    """
    with tf.variable_scope('layer1-parameters'):
        W1 = tf.get_variable("W1",[784,400],initializer=tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable("b1",[1,400],initializer=tf.zeros_initializer())
    
    with tf.variable_scope('layer2-parameters'):
        W2 = tf.get_variable("W2",shape=[400,784], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b2 = tf.get_variable("b2",shape=[1,784], initializer = tf.zeros_initializer())
    
# =============================================================================
#     with tf.variable_scope('layer3-parameters'):
#         W3 = tf.get_variable("W3",[100,400],initializer=tf.contrib.layers.xavier_initializer(seed=1))
#         b3 = tf.get_variable("b3",[1,400],initializer=tf.zeros_initializer())
#     
#     with tf.variable_scope('layer4-parameters'):
#         W4 = tf.get_variable("W4",[400,784],initializer=tf.contrib.layers.xavier_initializer(seed=1))
#         b4 = tf.get_variable("b4",[1,784],initializer=tf.zeros_initializer())
# =============================================================================

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2 }
# =============================================================================
#                   "W3": W3,
#                   "b3": b3,
#                   "W4": W4,
#                   "b4": b4 
# =============================================================================
    
    return parameters

# 前向传播函数
def forward_propagation(X,parameters):
    """
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
# =============================================================================
#     W3 = parameters["W3"]
#     b3 = parameters["b3"]
#     W4 = parameters["W4"]
#     b4 = parameters["b4"]
# =============================================================================

    
    with tf.variable_scope('layer1'):
        with tf.name_scope('XW_plus_b'):
            Z1 = tf.add(tf.matmul(X,W1),b1)
            #Z1 = tf.matmul(X,W1) + b1
        with tf.name_scope('relu'):
            A1 = tf.nn.relu(Z1)

    with tf.name_scope('layer2'):
        with tf.name_scope('XW_plus_b'):
            Z2 = tf.add(tf.matmul(A1,W2),b2)
        with tf.name_scope('relu'):
            A2 = tf.nn.relu(Z2)
            
# =============================================================================
#     with tf.name_scope('layer3'):
#         with tf.name_scope('XW_plus_b'):
#             Z3 = tf.add(tf.matmul(A2,W3),b3)
#         with tf.name_scope('relu'):
#             A3 = tf.nn.relu(Z3)
#     
#     with tf.name_scope('layer4'):
#         with tf.name_scope('XW_plus_b'):
#             Z4 = tf.add(tf.matmul(A3,W4),b4)
#         with tf.name_scope('relu'):
#             A4 = tf.nn.relu(Z4)
# =============================================================================
            
    return A2, A1
        
# 计算损失函数
def compute_cost(A2,Y):
    """
    """
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.square(Y - A2))
        #cost = tf.losses.mean_squared_error(Y,A2)
    return cost

# 反向传播函数
def backward_propagation(cost,learning_rate):
    """
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    return optimizer



# 添加噪声函数
def add_noise(data,sigma):
    """
    """
    # assert pixel value range is 0-1
    noise = np.random.normal(loc=0,scale=sigma/255,size=data.shape)
    data_noise = data + noise
    data_boise = np.clip(data_noise,0,1)
    
    return data_noise

# 计算信噪比函数
def cal_psnr(im1,im2):
    """
    """
    # assert pixel value range is 0-1
    mse = ((im1 * 255 - im2 * 255) ** 2).mean()
    psnr = 10 * np.log10(255 * 255 / mse)
    
    return psnr




def model(X_train, X_test, sigma = 15, Cal_Psnr_Sample_Num = 10, view_sample = 5, 
        learning_rate=0.001,num_epochs=10,minibatch_size=128,
        print_cost=True,isPlot=True, 
        display_iteration=200, display_epoch=1):
    """
    """
    tf.set_random_seed(1)
    (m,n_x) = X_train.shape            #获取输入样本数和节点数量
    costs = []                         #存储损失
    psnrs = []                         #存储峰值信噪比
    
    # 给X和Y创建placeholder
    X, Y = create_placeholders(n_x)
    
    # 初始化参数
    parameters = initialize_parameters()
    
    # 前向传播
    A2, A1 = forward_propagation(X,parameters)
    
    # 计算成本
    cost = compute_cost(A2,Y)
    
    # 反向传播,使用Adam优化
    optimizer = backward_propagation(cost,learning_rate)

    # 定义初始化所有的变量的python变量init(操作)
    init = tf.global_variables_initializer()
    
    
    # 定义一个计算图
    sess = tf.Session()
    # 将计算图保存到一个目录下
    #writer = tf.summary.FileWriter("logs/EasyDAE_model_logs/", sess.graph)
    # 初始化
    sess.run(init)
    
    print("\n"+"--------------------训练开始---------------------"+"\n")
    for epoch in range(num_epochs):
        
        epoch_cost = 0 #每代的成本预先定义量,mini_batch中使用
        num_minibatches = m // minibatch_size # mini_batch的总数量60000//128=468
        
        # 开始记录每一个epoch的训练时间参数
        start_time = time.clock()
        
        # 开始每一个iteration的训练
        for iteration in range(num_minibatches):
            
            # 选择一个minibatch
            minibatch = mnist.train.next_batch(minibatch_size)
            
            # 清晰图片加入噪声作为输入
            clear_image = minibatch[0]
            noise_image = add_noise(clear_image,sigma)
            
            
            # 数据已经准备好了，开始运行session
            _,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:noise_image,Y:clear_image})
            
            # 记录成本
            if iteration % 10 == 0:
                costs.append(minibatch_cost)
            
            #计算这个minibatch在这一代中所占的误差
            epoch_cost = epoch_cost + minibatch_cost / num_minibatches
        
        # 计算每一个epoch的PSNR
        clear_image = X_test[:Cal_Psnr_Sample_Num] # num_sample计算的平均PSNR的样本数量
        noise_image = add_noise(clear_image,sigma)
        reconstructed = sess.run(A2, feed_dict={X: noise_image})
        ## 将数据传入函数计算PSNR
        epoch_psnr = Cal_Epoch_AvgPsnr(reconstructed, noise_image)
        
        # 记录每一个epoch的PSNR
        psnrs.append(epoch_psnr)
        
        # 记录每一个epoch的时间
        end_time = time.clock()
        
        # 打印成本
        if print_cost:
            if epoch % display_epoch == 0:
                print("epoch: {0: >2d}/{1:d}, ".format(epoch+1,num_epochs),
                      "cost: {:.8f}, ".format(epoch_cost),
                      "Average PSNR: {:.2f}dB, ".format(epoch_psnr),
                      "Time: {:.2f}s".format(end_time - start_time))

# =============================================================================
#             # 打印成本（包含iteration的版本）
#             if print_cost:
#                 #每迭代100后打印成本
#                 if iteration % display_iteration == 0 and epoch % display_epoch == 0:
#                     print("epoch: {0: >2d}/{1:d}, ".format(epoch+1,num_epochs),
#                           "iteration: {:^3d}, ".format(iteration),
#                           "cost: {:.8f}".format(minibatch_cost)) 
#                 #打印每一个epoch经历的最大次数迭代时的成本
#                 if iteration == (num_minibatches-1) and epoch % display_epoch == 0:
#                     print("epoch: {0: >2d}/{1:d}, ".format(epoch+1,num_epochs),
#                           "iteration: {:^3d}, ".format(iteration),
#                           "cost: {:.8f}".format(minibatch_cost))
# =============================================================================
                
    print("\n"+"--------------------训练结束---------------------"+"\n")
    
    # 绘制成本曲线
    if isPlot:
        fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        axes1[0].plot(costs)
        axes1[0].set_ylabel("cost",fontproperties='SimHei',fontsize=15,color='black')
        axes1[0].set_xlabel("iteration (per tens)",fontproperties='SimHei',fontsize=15,color='black')
        axes1[0].set_title("learning rate = " + str(learning_rate),fontproperties='SimHei',fontsize=16,color='green')
        axes1[0].grid(True)
        axes1[1].plot(psnrs)
        axes1[1].set_ylabel("Average PSNR",fontproperties='SimHei',fontsize=15,color='black')
        axes1[1].set_xlabel("Epochs",fontproperties='SimHei',fontsize=15,color='black')
        axes1[1].set_title("$\sigma = %s$" %sigma, fontproperties='SimHei',fontsize=16,color='blue')
        axes1[1].grid(True)
    
    # 对测试图片测试
    test_image = X_test[:view_sample]
    # 清晰图片加入噪声作为输入
    test_image_noise = add_noise(test_image,sigma)
    # 噪声图片作为输入得到重构图片
    reconstructed, compressed = sess.run([A2, A1], feed_dict={X: test_image_noise})    
    
    # 存储测试结果
    test_image_dict = {}
    test_image_dict = {"test_image" : test_image, 
                 "test_image_noise": test_image_noise,
                 "reconstructed": reconstructed,
                 "compressed": compressed }
    
    
    return parameters,test_image_dict

def Cal_Epoch_AvgPsnr(image_array1,image_array2):
    """
    """
    num_sample = image_array1.shape[0]
    
    psnr_sum = 0
    for i in range(num_sample):
        psnr = cal_psnr(image_array1[i],image_array2[i])
        psnr_sum = psnr_sum + psnr
    average_psnr = psnr_sum / num_sample
    
    return average_psnr


# 查看测试图片的去噪效果
def test(test_image_dict):
    """
    """
    
    test_image = test_image_dict["test_image"]
    test_image_noise = test_image_dict["test_image_noise"]
    reconstructed = test_image_dict["reconstructed"]
    compressed = test_image_dict["compressed"]
    
    num_sample = test_image.shape[0] # 测试图片的样本数量
    # 绘图
    picture_size = 2 # 展示图片时的尺寸参数
    fig2, axes2 = plt.subplots(nrows=2, ncols=num_sample, 
                               sharex=True, sharey=True, 
                               figsize=(picture_size*num_sample,2*picture_size))  
    
    for images, row in zip([test_image_noise, reconstructed], axes2):
        for image, ax in zip(images, row):
            ax.imshow(image.reshape((28, 28)), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    fig2.tight_layout(pad=0.1)
    
    fig3, axes3 = plt.subplots(nrows=1, ncols=num_sample, 
                               sharex=True, sharey=True, 
                               figsize=(picture_size*num_sample,1*picture_size))
    for image, ax in zip(compressed, axes3):
        ax.imshow(image.reshape((20,20)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig3.tight_layout(pad=0)
    
    psnr_sum = 0
    for i in range(num_sample):
        psnr = cal_psnr(reconstructed[i],test_image_noise[i])
        psnr_sum = psnr_sum + psnr
    
    avg_psnr = psnr_sum / num_sample
    print(" Test Average PSNR : %.2fdB " % avg_psnr)
    
    return avg_psnr



# main
if __name__ == '__main__':
    
    # 制作训练、测试数据集
    X_train, X_test = classify_data(mnist)
    
    # 导入模型
    _, test_image_dict = model(X_train, X_test, sigma=15, num_epochs=50,
                               Cal_Psnr_Sample_Num=10, view_sample=5,  
                               minibatch_size=512, learning_rate=0.001, 
                               display_iteration=500, display_epoch=1)
    
    # 查看测试图片的去噪效果
    avg_panr = test(test_image_dict)


































