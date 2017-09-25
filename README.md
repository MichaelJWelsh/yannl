# Yannl
Yannl is a compact, portable, feed-forward artificial neural network library written in C++11. Yannl has no dependencies but can be easily optimized for matrix multiplication via callbacks (see integrating BLAS example for details).


## Features
* **Gradient Descent Algorithms**
    * batch
    * mini-batch
    * stochastic
* **Activation Functions**
    * sigmoid
    * relu
    * tanh
    * softmax
    * linear
* **Learning Rate Annealing**
* **L2 Regularization**
* **Momentum**
* **Fan-In Weight Initialization**
* **Serializability**


## Installation & Usage
To install, run following command from terminal :
```
sudo make
```
Use `LIB_PATH` and `INCLUDE_PATH` to install Yannl in a specific directory. By default, Yannl is installed at `/usr/local/lib` and header files are stored at `/usr/local/include`.


Include Yannl in your source via:
```C++
#include <yannl.hpp>
```
At compilation, use `-lyannl` flag. i.e.
```
g++ foo.cpp -lyannl -std=c++11
```

## Example
Use the Makefile in the `examples` folder to run each example. Below is the `realistic_classification` example. 


The neural network in this example is trained with a very large artificially generated training set. The goal is for it to be able to tell whether the sum of two numbers is positive or negative. The input is two floating-point numbers, and the output is either {1, 0} (positive), or {0, 1} (negative). After training, it is given seven sets of numbers and outputs a matrix classifying the sum of each set of numbers. In this case, the accuracy is 100%. Finally, the user can input their own data and see whether or not the neural network correctly classifies the sum. 


![](https://github.com/MichaelJWelsh/yannl/blob/master/example.gif)
