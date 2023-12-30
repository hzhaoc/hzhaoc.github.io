---
layout: post
title:  "principal of neural network"
date:   2023-12-29 22:00:00 -0800
brief: 'a physically-over-simplifed and engineering-complicated computer approximation of human mind'
---

## Inspiration
From human brains of neural nets\
![brain_neurons](/assets/images/brain_neurons.png)

## The binary property in the AI world
### Sigmoid Function

$$h_\theta(X)=g(\theta^TX)=\frac{1}{1+e^{-\theta^TX}}=\frac{1}{1+e^{-z}}=P(y=1|x)$$

\
where
\
$$z=\theta^TX$$
![sigmoid](/assets/images/sigmoid.png)

#### Cost Function
- $$J(\theta)=\frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(X^{(i)}), y^{(i)})$$
- $$J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}[y^{(i)} * log(h_\theta(X^{(i)})) + (1-y^{(i)}) * log(1-h_\theta(X^{(i)}))]$$

#### MLE (maximum likelihood estimation)
According to likelihood:
- $$L(\theta)=\prod_{i=1}^{m}P(y^{(i)}|X^{(i)})$$

obviously (Bernoulli Distribution)
- $$L(\theta)=\prod_{i=1}^{m}[h_{\theta}(X^{(i)})^{y^{(i)}}(1-h_{\theta}(X^{(i)}))^{1-y^{(i)}}]$$

log both sides:
- $$l(\theta)=\sum_{i=1}^{m}[y^{(i)}*log(h_{\theta}(X^{(i)})) + (1-y^{(i)})*log(1-h_{\theta}(X^{(i)}))]$$

total cost is to maximize 
- $$l(\theta)$$ 

\
or to minimize
- $$J(\theta)=-\frac{1}{m}l(\theta)$$

#### Gradient Descent
if negative log-likelihood is: (just a different form from above $J(\theta)$)
- $$NLL\left (D, \mathbf{w} \right ) = -\sum_{i=1}^{N} \left [ \left ( 1 - y_i \right ) \log(1-\sigma(\mathbf{w}^T\mathbf{x}_i)) + y_i\log \sigma(\mathbf{w}^T\mathbf{x}_i)  \right ]$$

gradient descent process is 
- $$\mathbf{w_t} = \mathbf{w_{t-1}} - \eta\frac{\mathrm{d}NLL(D, \mathbf{w})}{\mathrm{d}\mathbf{w}}$$

gradient is 
- $$\eta\frac{\mathrm{d}NLL(D, \mathbf{w})}{\mathrm{d}\mathbf{w}}$$

where 
- $\eta$ is the learning rate

then $\forall\mathbf{w}_j \in [1, 2, ..., m]$:\
\
$$\frac{\partial NLL(D, W)}{\partial{W}_j}= -\frac{\partial\sum_{i=1}^{n}[(1-y_i)log(1-\sigma(W^TX_i)) + y_ilog\sigma(W^TX_i)]}{\partial{W}_j}$$
\
$$\frac{\partial NLL(D, \mathbf{W})}{\partial\mathbf{W}_j}= -\sum_{i=1}^{n}[(y_i)\frac{1}{\sigma(\mathbf{W}^TX_i)}\frac{\partial\sigma(\mathbf{W}^TX_i)}{\partial\mathbf{W_j}} - (1-y_i)\frac{1}{1-\sigma(\mathbf{W}^TX_i)}\frac{\partial\sigma(\mathbf{W}^TX_i)}{\partial\mathbf{W_j}}]$$
\
$$\frac{\partial NLL(D, \mathbf{W})}{\partial\mathbf{W}_j}= -\sum_{i=1}^{n}[(y_i)\frac{1}{\sigma(\mathbf{W}^TX_i)} - (1-y_i)\frac{1}{1-\sigma(\mathbf{W}^TX_i)}]\frac{\partial\sigma(\mathbf{W}^TX_i)}{\partial\mathbf{W_j}}$$
\
$$\frac{\partial NLL(D, \mathbf{W})}{\partial\mathbf{W}_j}= -\sum_{i=1}^{n}[(y_i)\frac{1}{\sigma(\mathbf{W}^TX_i)} - (1-y_i)\frac{1}{1-\sigma(\mathbf{W}^TX_i)}]\sigma(\mathbf{W}^TX_i)(1-\sigma(\mathbf{W}^TX_i))\frac{\partial\mathbf{W}^TX}{\partial\mathbf{W_j}}$$
\
$$\frac{\partial NLL(D, \mathbf{W})}{\partial\mathbf{W}_j}= -\sum_{i=1}^{n}[(y_i)(1-\sigma(\mathbf{W}^TX_i)) - (1-y_i)(\sigma(\mathbf{W}^TX_i))]X_i^j$$
\
$$\frac{\partial NLL(D, \mathbf{W})}{\partial\mathbf{W}_j}= -\sum_{i=1}^{n}[y_i - \sigma(\mathbf{W}^TX_i)]X_i^j$$
\
$$\frac{\partial NLL(D, \mathbf{W})}{\partial\mathbf{W}_j}= \sum_{i=1}^{n}[\sigma(\mathbf{W}^TX_i) - y_i]X_i^j$$
\

where 
- $X_i^j$ denotes $j$th feature of the $i$th patient training vector.
- $\mathbf{W}$ is coefficient vector.

#### Stochastic Gradient Descent (SGD)
update $\mathbf{w_t}$ based on a single pair of $X, y$ , SGD is:
- $$\mathbf{w_{t}} = \mathbf{w_{t-1}} - \eta(\sigma(\mathbf{w}^T\mathbf{x}_t) - y_t)\mathbf{x}_t$$

with L2 Norm regularization ($J= NLL + \mu\Vert{\mathbf{W}}\Vert_2^2$), SGD is:
- $$\mathbf{w_{t}} = (1-2\eta\mu)\mathbf{w_{t-1}} - \eta(\sigma(\mathbf{w}^T\mathbf{x}_t) - y_t)\mathbf{x}_t$$


## Neural Network Basic Form
A basic structure of neural network is 1 input layer, certain number of hidden layers, and 1 output layer. For each arrow in the below illustration example, it's an activation function, for example, one popular activation function is *Logistic Regression*; each activation neuron in one layer is an output from from each activation function with last layer's activation neurons.\
![simple_nn](/assets/images/neural_network.png)

## Neural Network Forward Propagation
In the above example, each layer of activation neurons (input can be viewed as the $0_{th}$ layer of activation neuron) is an input to each logistic regression to output next layer of neurons, until the final output of binary or multiclass.

Formally, the forward propagation does the following:
- $a_1^{(2)}=g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3)$
- $a_2^{(2)}=g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3)$
- $a_3^{(2)}=g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3)$
- $h_\Theta(x)=g(\Theta_{10}^{(2)}a_0^{2} + \Theta_{11}^{(2)}a_1^{2} + \Theta_{12}^{(2)}a_2^{2} + \Theta_{13}^{(2)}a_3^{2})$

\
where
- $a_{i}^{(j)}$ is $j_{th}$ layer $i_{th}$ activation neuron;
- $\Theta_{ij}^{(k)}$ is $k_{th}$ layer $i_{th}$ neuron $j_{th}$ coefficient, in the illustration example, $\Theta_{0j}^{(k)}$ is bias / intercept term and $x_0=0$;
- $h_\Theta(x)$ is the final output, here it's binary 0  or 1.

#### Cost Function
We know each activation function in the network is a single Logistic Regression who input from last layer neurons, and output to next layer neurons. 

**One form** of cost function, for a multiclass neural network, total cost including L2 Norm is:
- $$J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}[y_k^{(i)}log(h_\Theta(x^{(i)}))_k + (1-y_k^{(i)})log(1-log(h_\Theta(x^{(i)}))_k)] + \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{ji}^{(l)})^2$$

\
where
- $K$ is number of classes in output layer (including input and output)
- $h_\Theta(x^{(i)})_k$ is the **whole** binary-output function on the $k$-th output class, for the $i$-th training example. (**Not just logistic regression to output to last layer**)

**Another form** of cost function without regularization, is as below:
- $$J(\theta)=\frac{1}{2m}\sum_{i}^{m}\sum_{k}^{K}(h_\Theta(x^{(i)})_k-y^{(i)}_k)^2$$

\
Same notations.

## Backward Propagation
#### Notation
![neural_network2.png](/assets/images/neural_network2.png)

Given above 4-layered NN structure, following previous notation customs, we make following notations:
- $a^{(1)}=x$
- $z^{(2)}=\Theta^{(1)}a^{(1)}$
- $a^{(2)}=g(z(^{(2)}))$
- $z^{(3)}=\Theta^{(2)}a^{(2)}$
- $a^{(3)}=g(z(^{(3)}))$
- $z^{(4)}=\Theta^{(3)}a^{(3)}$
- $a^{(4)}=g(z(^{(4)}))=h_\Theta(x)$

### Gradient Computation
#### 'Error' term
Intuition: $\delta_j^{(l)}$ is the 'error' of node $j$ in layer $l$.

For each output unit ($.*$ is element wise vector multiplication, here are all vector form):
- $\delta^{(4)}=a^{(4)}-y$
- $\delta^{(3)}=(\Theta^{(3)})^T\delta^{(4)}.*g'(z^{(3)})$ 
  - where $g'(z^{(3)})=a^{(3)}.*(1-a^{(3)})$ for logistic regression
- $\delta^{(2)}=(\Theta^{(2)})^T\delta^{(3)}.*g'(z^{(2)})$
- No $\delta^{(1)}$


#### Gradient Derivation [^1]
1. $$\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=a_j^{(l)}\delta_i^{(l+1)}$$ 

derivation (ignore regularization):

$\frac{\partial{L}}{\partial{\Theta_{ij}^{(l)}}}=\frac{\partial{L}}{\partial{z_{i}^{(l)}}}\frac{\partial{z_{i}^{(l)}}}{\partial{\Theta_{ij}^{(l)}}}=\delta_i^{(l)}a_j^{(l-1)}$


2. $$\delta_{j}^{(l)}=\frac{\partial}{\partial{z_{j}^{(l)}}}cost(i)=[(\delta^{(l+1)})^T\Theta_j^{(l)}]g'(z_j^l)$$

derivation:
- $\delta_i^{(l)}=\frac{\partial{J}}{\partial{z_i^{(l)}}}$
- $=\sum_k\frac{\partial{J}}{\partial{z_k^{(l+1)}}}\frac{\partial{z_k^{(l+1)}}}{\partial{z_i^{(l)}}}$
- $=\sum_k\delta_k^{(l+1)}\Theta_{kj}^{(l)}g'(z_j^l)$
- $=[(\delta^{(l+1)})^T\Theta_j^{(l)}]g'(z_j^l)$ where $\delta$, $\Theta_j$ here denotes horizontal vector, vertical vector. 

**Be careful with the sum here because of the sum of error forms in Cost Function**

Thus vectorwise we have:
- $\delta^{(l)}=(\Theta^{(l)})^T\delta^{(l+1)}.*g'(z^{(l)})$
\
where $z$ is activation function for each neuron / unit

#### Backward propagation computation process
Set $\Delta_{ij}^{(l)}=0$ as error vectors for all layers
For $i=1$ to $m$:
- Set $a^{(1)}=x^{i}$
- Perform forward propagation to compute $a^{(l)}$ for all layers all neurons
- Perform backward propagation to compute $\partial$ for all layers all neurons
- $\Delta_{ij}^{(l)}:=\Delta_{ij}^{(l)} + a_j^{(l)}\delta_i^{(l+1)}$

Finally:
- $D_{ij}^{(l)}:=\frac{1}{m}\Delta_{ij}^{(l)} + \lambda\Theta_{ij}^{(l)}$ if $j\neq0$ (contains regularization term)
- $D_{ij}^{(l)}:=\frac{1}{m}\Delta_{ij}^{(l)}$  if  $j = 0$ (bias term)
- $\frac{\partial}{\partial\Theta_{ij}^{(l)}}J(\Theta)=D_{ij}^{(l)}$ <- now you have the gradient

#### BP Intuition
![nn_bp](/assets/images/nn_bp.png)

Similar intuition from FP, $\delta_{j}^{(l)}$ is the 'error' of cost for $a_{j}^{(l)}$; it's weighted sum from this layer's corresponding $\theta s$ and latter layer's 'errors' multiplied by first derivative of activation function. The gradient for each $\theta$ is the product of latter layer's corresponding error, and this layer's corresponding activation unit value.

The key idea of gradient descent to optimize $\Theta$ in neural network for $m$ training samples, is to **understand error term and gradient in backward propagation**

#### Random Initialization
Since if $\Theta=0$, gradient update will also be the same, $\Theta$ needs to initialize $\Theta$ with random small values.


[^1]: [Michael Nielsen, *Neural Networks and Deep Learning, 2.4, 2.5*](http://neuralnetworksanddeeplearning.com/)
    