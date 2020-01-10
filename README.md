# DNN
### Deep (Vanilla) Neural Network for classifying hand-written digits, written in C++
This program classifies the MNIST hand-written digit dataset, with 60000 training samples and 10000 testing samples, all in 28x28 grayscale resolution. This is a classic problem, and there have been numerous methods published for it: http://yann.lecun.com/exdb/mnist/. One of the pioneers of MNIST, Yann LeCun, is renowned in the field of machine learning. The whole network is hard-coded, only using libraries for basic tools such as square root, exponential functions, and std::vector. 
### Details & Parameters
My program is a vanilla neural network, with an arbitrary number of hidden layers. So far I have been sticking with two layers, which is fairly typical. You can also choose how many nodes are in each hidden layer, but the input layer and decision layer (of course) have 784 and 10 nodes, respectively. Connections only exist from one layer to the one directly after it, and adjacent layers are fully connected. Each node in the hidden layer has a bias, and an activation function you can choose. So far I've worked with a logistic activation function with adjustable steepness and range [0,1]. In the future, I will add ReLU. 

One of the most pivotal parameters is the learning rate. I'm using a changing learning rate (a schedule) for better results. So far I've tried using a constant rate, exponentially decaying rate, and linearly decaying rate. The linear method works the best: it starts at a high rate and decreases linearly until a certain lower bound; the initial value, decay slope, and lower bound are all controlled with constants in the code.

Another parameter is the batch size. Currently it's set at 1 for stochastic gradient descent. Also, using more than 15000 of the training sample leads to some memory allocation issues, and as soon as I sort that out I will mess with mini-batch gradient descent. 

You must run the program in the same direcoty as all of the data files. You can find them on the MNIST website in zipped form as well.

### Results
1/10/20: ~27%. This is quite high, but the structure is non-optimal: 784-35-35-10, logistic activation, SGD, and linear learning schedule.<br/>
1/10/20: 19.28%. Same structure as above, but fixing memory leaks allows using **all** 60000 training samples.
