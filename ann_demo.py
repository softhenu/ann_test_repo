import numpy as np
import scipy.special
import scipy.misc
import matplotlib.pyplot
import scipy.ndimage

class NeuralNetwork():
    #初始化神经网络
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.onodes=outputnodes
        self.hnodes=hiddennodes
        self.lr=learningrate
        #权重矩阵设置，正态分布
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #激活函数设置
        self.activation_function=lambda x:scipy.special.expit(x)
    #训练神经网络
    def train(self,input_list,target_list):
        inputs=np.array(input_list,ndmin=2).T
        targets=np.array(target_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_inputs)
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors)
        #隐藏层和输出层权重更新
        self.who+=self.lr*np.dot(output_errors*final_outputs*(1-final_outputs),np.transpose(hidden_outputs))
        self.whi+=self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),np.transpose(inputs))
        pass
#我这边只进行测试罢了