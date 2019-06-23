#-*- coding:utf-8 -*-
import numpy as np
import scipy.special
class ANFIS_System:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #设置节点的输入，输出，隐藏层
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 链接权重矩阵
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #learning rate 学习率
        self.lr = learningrate

        #激活函数 activation function is the sigmoid function
        self.activation_function = lambda  x:scipy.special.expit(x)
        pass
    def train(self, inputs_list, targets_list):#训练模型train the model
        #将输入list转为2维矩阵
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T

        #calculate signals into hidden layer  计算隐藏层输入信号
        hidden_inputs = np.dot(self.wih , inputs)

        #calculate the signals emerging from hidden layer# 计算 隐藏层输出信号
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into final output layer 计算输入输出层信号
        final_inputs = np.dot(self.who , hidden_outputs)

        #calculate the signals emerging from final output layer 计算输出层输出
        final_outputs = self.activation_function(final_inputs)

        #output layer error is the (target-actual)计算输出层误差
        output_errors = targets - final_outputs

        #hidden layer error is the output_errors,split by weights recombined at hidden nodes计算隐藏层误差
        hidden_errors = np.dot(self.who.T , output_errors)

        #update the weights for the links between the hidden and output layers更新隐藏层和输出层的权重
        self.who += self.lr * np.dot((output_errors*final_outputs*(1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layers更新输入层和隐藏层的权重
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs*(1.0 -hidden_outputs)),
                                      np.transpose(inputs))

        pass
    def query(self, inputs_list):#query the model query()函数接受神经网络的输入，返回网络的输出。这个功能非常简 单，但是，为了做到这一点，你要记住，我们需要传递来自输入层节点的
#输入信号，通过隐藏层，最后从输出层输出。
#query只需要input_list不需要其他任何输入
        #将输入list转换成二维数组
        inputs = np.array(inputs_list,ndmin=2).T
        #calculate signals into hidden_layer
        hidden_inputs = np.dot(self.wih,inputs)
#calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

#calculate signals into final output layer
        final_inputs = np.dot(self.who,hidden_outputs)
#calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return  final_outputs



input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#learning rate setting
learning_rate = 0.5

#create instance of neural network
n=ANFIS_System(input_nodes,hidden_nodes,output_nodes,learning_rate)
print(n.query([1.0,0.5,-1.5]))
