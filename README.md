# neural_network

## Abstract
We built a powerful and flexible neural network simulator framework from scratch. We also provided complete tools for hyper-parameter tuning as well as validation and assessment of the model performance.

## Method
The implemented MLP relies heavily on Numpy, the python library optimized for scientific calculations. The reason behind this choice is that the neural network we made do its operations using only vectors and matrices; thanks to this design choice we were able to make elegant code and exploit the fact that Numpy is optimized for such structures (so enhancing the performances of our model). 

The MLP we present here comes with the following features:

*  Possibility to define an arbitrary number of hidden layers, each one formed by an arbitrary number of units.
*  The most common and known activation functions, such as linear, sigmoid, relu and tanh; we also made possible to define easily new activation functions (provided the function and its derivative).
*  Stochastic Gradient Descent (SGD) and its variant with Nesterov Momentum.
*  Possibility to perform batch, online or mini-batch training.
*  The most common layer weight initializers, using known probability distributions such as uniform and Gaussian.
*  Tikhonov regularization for the weights.
*  An early stopping framework, which allows to prematurely stop the training by monitoring a specified quantity. Moreover, this feature comes with the possibility to make the user specify parameters such as the patience and the tolerance of the improvements. 
*  Automated and parallelized (multiprocessing) grid and random search, with rich logic rules for hyper-parameter generation.
*  All major validation and assessment procedures i.e. K-fold, Double K-fold and hold-out are automated and can be combined with the grid search.


Among the novelties has been implemented an autodiff framework able to mirror basic Numpy operations keeping track of the gradient.
