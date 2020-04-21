# -*- coding: utf-8 -*-
# @Time      : 11:41 AM
# @ Author   : Xiaojuan
# @CNblogs   :
print("Welcome to Tensorflow!")

# linear regression
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#   消元方法--对应最小二乘法
#   计算mse
def mse(b,w,points):
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (w*x+b))**2
    # 返回均方差
    return totalError/float(len(points))

#   梯度下降方法
def stepGradient(b_current,w_current,points,lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        #   计算误差函数--b,w求导
        b_gradient += (2/M)*((w_current*x+b_current)-y)
        w_gradient +=(2/M)*x*((w_current*x+b_current)-y)
        #   更新梯度值
        new_b = b_current - (lr * b_gradient)
        new_w = w_current - (lr * w_gradient)
        return [new_b,new_w]

#   梯度下降——循环
def gradient_descent(points,start_b,start_w,lr,num_iteration):
    b = start_b
    w = start_w
    for step in range(num_iteration):
        b,w = stepGradient(b,w,np.array(points),lr)
        loss = mse(b,w,points)
        if step%50 == 0:
            print(f"inteation:{step},loss:{loss},w:{w},b:{b}")
    return [b,w]


if __name__ == '__main__':

    #   创建随机数据
    data = []
    for i in range(100):
        x = np.random.uniform(-10.0,10.0)
        # 高斯采样
        eps = np.random.normal(0.,0.01)
        y = 1.477 * x + 0.089 + eps
        data.append([x,y])
    data = np.array(data)
    #   梯度递归
    lr = 0.01
    init_b = 0
    init_w = 0
    num_iter = 1000
    [b,w] = gradient_descent(data,init_b,init_w,lr,num_iter)
    # loss = mse(0.089,1.477,data)
    # print("The Result is ",loss)
    loss = mse(b,w,data)
    print(f"Final loss:{loss},w:{w},b{b}")

    print("All finished!")




