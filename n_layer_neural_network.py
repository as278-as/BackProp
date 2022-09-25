from ctypes import sizeof
from turtle import clear
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

from BP import NeuralNetwork
from Layer import Layer

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


class n_layer_neural_network(NeuralNetwork,Layer):

    def __init__(self, layers_dim,actFun_type='Tanh', reg_lambda=0.01, seed=0):
        self.layers_dim=layers_dim #list of dimensions of each layer
        self.initializeParams(layers_dim); 
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda


    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.N_Layer_FeedForward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.Y_pred, axis=1)
    

    def initializeParams(self,layers_dims):
        self.W={};
        self.B={};
        for i in range (1,len(layers_dims)):
            self.W["W"+str(i)]=np.random.randn(layers_dims[i-1],layers_dims[i])/ np.sqrt(layers_dims[i]);
            self.B["B"+str(i)]=np.zeros((1,layers_dims[i]));

    def N_Layer_FeedForward(self, X, actFun):
        A_prev=X.T; #dimensions of X are 200x2, I am changing to 2x200
        self.mem=[];
        for i in range(1,len(self.layers_dim)):
            # print(i)
            Z,layer_mem=self.single_layer_feedForward(A_prev,self.W["W"+str(i)],self.B["B"+str(i)]);
            self.mem.append(layer_mem);
            if i==len(self.layers_dim)-1:
                A[A>=100]=0;
                A=np.exp(Z);
                A = A / np.sum(A, axis=0, keepdims=True);
            else :
                A=actFun(Z);
            A_prev=A;
        
        self.Y_pred=A.T;#to make dimensions 200x2


    def N_Layer_BackProp(self, X, y):
        self.delta={};
        num_examples = len(X);
        deltaN=self.Y_pred;
        deltaN[range(num_examples), y] -= 1;
        self.delta["D"+str(len(self.layers_dim))]=deltaN;
        self.grad_params={};
        for i in range(len(self.layers_dim)-1,1,-1):
            if i==len(self.layers_dim)-1:
                Z_1,A_prev_1,W_1,B_1=self.mem[i-2];
                self.delta["D"+str(i)]=(self.W["W"+str(i)]@self.delta["D"+str(i+1)].T)*self.diff_actFun(Z_1,self.actFun_type);
                
            else:
                self.delta["D"+str(i)]=self.single_layer_BackProp(self.W["W"+str(i)],self.delta["D"+str(i+1)],self.mem[i-2],self.actFun_type);
        
        for i in range(len(self.layers_dim)-1,0,-1):
            Z,A_prev,W,B=self.mem[i-1];
            # self.delta["D"+str(i)]=self.single_layer_BackProp(self.W["W"+str(i)],self.delta["D"+str(i+1)],self.mem[i-2],self.actFun_type);
            if i==len(self.layers_dim)-1:
                self.grad_params["DW"+str(i)]=A_prev@self.delta["D"+str(i+1)];
                self.grad_params["DB"+str(i)]=np.sum(self.delta["D"+str(i+1)],axis=0,keepdims=True);
            else:
                # tmem= i-2 if i>=2 else 0
                Z_1,A_prev_1,W_1,B_1=self.mem[i-1];
                # self.delta["D"+str(i)]=(self.W["W"+str(i)]@self.delta["D"+str(i+1)])*self.diff_actFun(Z_1,self.actFun_type);
                self.grad_params["DW"+str(i)]=A_prev_1@self.delta["D"+str(i+1)].T;
                self.grad_params["DB"+str(i)]=np.sum(self.delta["D"+str(i+1)].T,axis=0,keepdims=True);



    def fit_model(self, X, y, epsilon=0.005, num_passes=50000, print_loss=True):
        # Gradient descent.
        for j in range(0, num_passes):
            # Forward propagation
            self.N_Layer_FeedForward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            self.N_Layer_BackProp(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for i in range(1,len(self.layers_dim)):
                self.grad_params["DW"+str(i)]+=self.reg_lambda*self.grad_params["DW"+str(i)];


            # Gradient descent parameter update
            for i in range(1,len(self.layers_dim)):
                self.W["W"+str(i)]+=-epsilon *self.grad_params["DW"+str(i)];
                self.B["B"+str(i)]+=-epsilon *self.grad_params["DB"+str(i)];

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and j % 1000 == 0:
                print("Loss after iteration %i: %f" % (j, self.calculate_loss2(X, y)))


    def calculate_loss2(self, X, y):
            '''
            calculate_loss computes the loss for prediction
            :param X: input data
            :param y: given labels
            :return: the loss for prediction
            '''
            num_examples = len(X)
            self.N_Layer_FeedForward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Calculating the loss

            # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
            b = np.zeros((y.size, y.max() + 1));
            b[np.arange(y.size), y] = 1
            data_loss = -1*np.sum(b*np.log10(self.Y_pred));

            # Add regulatization term to loss (optional)
            # data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
            return (1. / num_examples) * data_loss

def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data();
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    model=n_layer_neural_network([2,3,5,4,2],actFun_type='Tanh');
    model.fit_model(X,y)
    model.visualize_decision_boundary(X,y);

if __name__ == "__main__":
    main()