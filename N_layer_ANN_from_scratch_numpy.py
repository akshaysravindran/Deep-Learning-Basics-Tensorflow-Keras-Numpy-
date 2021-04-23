
"""
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                        %
%                Create a Neural Network from scratch in numpy           %
%                                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  Summary: This script is used to create a N layer neural network designed 
  from scratch in numpy

 Credits: This script is made possible due to the guidance from tan_nguyen 
 the TA for the introduction to deep learning course at the Rice University 
 Author: Akshay Sujatha Ravindran
 email: akshay dot s dot ravindran at gmail dot com
 Dec 23rd 2018
"""


import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(1000, noise=0.20)
    return X, y

def generate_circle_data():
    '''
    generate circle data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(1000, noise=0.01)
#    data=datasets.load_iris()
#    X=data.data
#    y=data.target
    return X, y


def plot_decision_boundary(pred_func, X, y,title):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    
    # If multidimensional input
#    pca = decomposition.PCA(n_components=2)
#    pca.fit(X)
#    X = pca.transform(X)

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
    
    
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(X[:, 0], X[:, 1],X[:,2], s=40, c=y, cmap=plt.cm.Spectral)



    f, (ax1, ax2) = plt.subplots(1, 2,sharey=True)
    ax1.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    ax1.set_title('Original points')
    
    ax2.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    ax2.set_title(title)
    
    
    
    plt.savefig(title+'.png', bbox_inches='tight')


    plt.show()



class Layer(object):
    def __init__(self, nn_input_dim, nn_output_dim, last_layer_flag = 0, actFun_type='tanh', reg_lambda=0.01,seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.last_layer = last_layer_flag

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.nn_input_dim, self.nn_output_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_output_dim))


    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''
        
        # YOU IMPLEMENT YOUR feedforward HERE
        self.z = np.dot(X, self.W) + self.b

        # Apply the activation function specified by act_Fun_type for all layers except the output layer:
        if self.last_layer == 0:
            self.a = actFun(self.z)
        # Apply softmax to the last Layer:
        else:
            exp_scores = np.exp(self.z)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return None



class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, architecture, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.architecture = architecture
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.num_layers = np.shape(self.architecture)[0]
        

        # Build the network
        self.Build_Layer = []        
        layer=range(self.num_layers - 1)
        layer=[x+1 for x in layer]
        flag_last_layer=list(np.repeat(0,self.num_layers - 2))
        flag_last_layer.append(1)
        for ID,OD,LL_flag,s in zip(self.architecture[:-1],self.architecture[1:],flag_last_layer,layer):        
            x = Layer(nn_input_dim=ID, nn_output_dim=OD, last_layer_flag=LL_flag, actFun_type=self.actFun_type, seed = s) 
            self.Build_Layer.append(x)           
        



    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        if type == 'Tanh':
            activation = np.tanh(z)

        elif type == 'Sigmoid':
            activation = 1 / (1 + np.exp(-z))

        elif type == 'ReLU':
              activation = z * (z > 0)
        else:
            raise Exception('Activation Function is not supported!')

        return activation

    def diff_actFun(self, z, type):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        if type == 'Tanh':
            act_derivative = (1 - np.tanh(z)**2)

        elif type == 'Sigmoid':
            value = 1 / (1 + np.exp(-z))
            act_derivative = value * (1 - value)

        elif type == 'ReLU':
            act_derivative = (z > 0) * 1

        else:
            raise Exception('Activation Function is not supported!')

        return act_derivative

    def feedforward(self, X):
        '''
        feedforward does the forward pass of the n-layer Multi layer perceptron and computes the two probabilities,
        one for class 0 and the other one for class 1
        :param X: input data        
        :return: None
        '''

        for layer in range(self.num_layers - 1):
            if layer == 0:
            #For Input layer
              self.Build_Layer[layer].feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            elif (layer < self.num_layers - 2):
            #For Hidden layers
              self.Build_Layer[layer].feedforward(self.Build_Layer[layer-1].a, lambda x: self.actFun(x, type=self.actFun_type))
            else:
            #For Output layer
              self.Build_Layer[layer].feedforward(self.Build_Layer[layer-1].a, lambda x: self.actFun(x, type=self.actFun_type))
              self.probs = self.Build_Layer[layer].probs

        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X)
        # Calculating the loss


        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))

        # Add regulatization term to loss
        tempsum = 0
        for layer in range(self.num_layers - 1):
            tempsum += np.sum(np.square(self.Build_Layer[layer].W))
        data_loss += self.reg_lambda / 2 * tempsum

        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        dW = []
        db = []
        delta = []
        for layer in range(self.num_layers-1):
            dW.append([])
            db.append([])
            delta.append([])
        num_examples = len(X)
        # Delta for the last layer
        delta[self.num_layers-2] = self.probs
        delta[self.num_layers-2][range(num_examples), y] -= 1
        
        for layer in range(self.num_layers-2, -1, -1): 
            if layer != self.num_layers-2:
                delta[layer] = np.dot(delta[layer + 1],(self.Build_Layer[layer + 1].W.T))  * self.diff_actFun(self.Build_Layer[layer].z, self.actFun_type)
            if layer==0:
                dW[layer] = (X.T).dot(delta[layer])
            else:
                dW[layer] = (self.Build_Layer[layer - 1].a.T).dot(delta[layer])
            db[layer] = np.sum(delta[layer], axis=0, keepdims=True)

        return dW, db


    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
            
        '''
        loss=[]
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for layer in range(self.num_layers -1):
                dW[layer] += self.reg_lambda * self.Build_Layer[layer].W
                self.Build_Layer[layer].W += -epsilon * dW[layer]
                self.Build_Layer[layer].b += -epsilon * db[layer]


            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 500 == 0:
                loss.append(self.calculate_loss(X, y))
                print("Loss after iteration %i: %f" % (i, loss[-1]))
        return loss   

    def visualize_decision_boundary(self, X, y,title):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y,title)


def colormap_new():
    tableau20 = [(31, 119, 180), (174, 109, 232), (255, 127, 14), (255, 117, 120),  
                     (44, 160, 44), (112, 223, 138), (214, 39, 40), (255, 152, 150),  
                     (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                     (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
        (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.)
        
    return tableau20
        
        
        
# def main():
    
tableau20          = colormap_new()
X, y               = generate_circle_data()    # Generate the data
loss, unit, STRING = [], [], []
activation_function= ["ReLU","Sigmoid","Tanh"] # different types of activation functions
# Describe you architecure below, mention the nodes in different layers separted by ','
#for e.g. in architecture=[3,8,5,5,8,2], 3 is the nuber of input nodes, 2 is the output nodes
# and the in between numbers corresponds to the hidden layer units

Arch={"a1":[2,5,2],"a2":[2,8,5,8,2],"a4":[2,10,20,10,5,2],"a5":[2,200,185,100,85,30,10,2]}
   

for architecture in Arch.values():
    for actf in activation_function:    
        model       = DeepNeuralNetwork( architecture,actFun_type=actf)
        loss.append(model.fit_model(X, y, epsilon = 0.001))  
        string_out  =actf+"_architecure_"+" ".join(str(x) for x in architecture)
        STRING.append(string_out)
        model.visualize_decision_boundary(X,y,string_out)


plt.figure()
for i in range(12):
    plt.plot(loss[i],lw='3',color=tableau20[i],label=STRING[i])
    plt.legend()
    plt.xlabel('Iterations',fontsize=16)
    plt.ylabel('Loss',fontsize=16)    
plt.show()
plt.savefig('Loss_new.png', bbox_inches='tight') 

# if __name__ == "__main__":
#     main()
