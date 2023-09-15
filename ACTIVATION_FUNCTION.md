# Activation Functions
ML/DL make use of linearly weighted combinations of trainable parameters (weights) and input values (features). Linear combinations though will always be linear and hence are not suitable for updating the parameters of the model in the optimization phase. Therefore, non-linear activation functions are necessary to generate a statistical model through training it. Each linear combination will be processed by a non-linear activation function, which makes the model weights *trainable*. Commonly, activation functions are chosen that change their behavior for values smaller than 0 and vice versa. (https://www.academia.edu/download/69679997/29-485.pdf)

## Sigmoid
The sigmoid or logistic function is an s-shaped non-linear function. It ranges between 0 and 1, which makes it ideal to represent probabilities. On the other hand it saturates for very large and very small values, which can cause to some problems during the optimization.

$$
f(x)=\frac{1}{1+e^{-x}}
$$

## Hyperbolic Tangent
The bipolar sigmoid or hyperbolic tangent is similar in characteristic as the sigmoid function. It ranges from -1 to 1 and also introduces a threshold for large absolute values. The hyperbolic tangent is approximately linear for small absolute values.

$$
f(x)=\frac{e^{2*x}-1}{e^{2*x}+1}
$$

## ReLU
The rectified linear unit is a very simple function that takes the max between 0 and the input value, which maps every input value smaller than 0 to 0 and any input value greater than 0 linearly. This makes the derivative of the function very easy.

$$
f(x)=max(0,x)
$$

## Leaky ReLU
The leaky rectified linear unit is similar to ReLU but adds a light slope to the input values smaller than 0. This method is used to avoid getting dead units in the network when the gradient becomes zero and no update will be performed anymore (which was the case in ReLU). The $\alpha$ value determines the slope and is usually set to 0.1.

$$
f(x)=max(\alpha x,x)
$$

## ELU
The exponential linear unit is similar in concept to ReLU and leaky ReLU but makes the function also differentiable at 0 at the cost of a more complex computation since it involves the exponential function. The $\alpha$ value is set to a value between 0 and 1.

$$
f(x)=\begin{cases}
        x & x \ge 0 \\ 
        \alpha (e^{x} - 1 ) & x < 0
     \end{cases}
$$