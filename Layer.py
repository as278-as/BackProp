from ctypes import sizeof
from turtle import clear
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

class Layer:

    def __init__(self):
        print("jh")

    def single_layer_feedForward(self,A_prev,W,B):
        Z=W.T@A_prev+B.T;
        mem=(Z,A_prev,W,B);
        return Z,mem;

    def single_layer_BackProp(self,W,D,mem,actFun_type):
        Z_1,A_prev_1,W_1,B_1=mem;
        Z=self.diff_actFun(Z_1,actFun_type);
        return (W@D)*Z;