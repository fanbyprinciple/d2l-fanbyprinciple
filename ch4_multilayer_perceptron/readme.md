# Ch 4 multi layer perceptron

- to know about overfitting , underfitting
- regularisation
- creating deep neural networks more than one layer big

# Multi layer perceptron

- Hidden layers - if only one layer of affine solutions is needed for describing the model, then that would be sufficient.
- why activations
![](why_activations.png)

We need activations in between because otherwise the layers would collapse into a linear transformation.

with activations we have,

`H = sigma(XW1 + B1)`
`O = HW2 + B2`

# universal approximators

- a single hidden layer can approximateany function,however in practice we use many hidden layers.
- activation function -
Activation function decides whether a neuron should be activated or not by calculating the weighted sum and adding a bias to it. Differntiable operators that adds nonlinearity.

### Relu activation 

Rectified linear unit, given X, `relu(X) = max(0,X)`

![](relu.png)

There is also one parameterised relu, `prelu(x) = max(0,x) + alpha * min(0,x) `

Relu has simple derivative function which lets some of th eparameter through or just vanish which is very useful for dealing with problem of vanishing gradients.

### Sigmoid function

- sigmoid function squashes the variable range which lies in between (-inf, inf) to (0,1), thus it is also called squashing function
- `sigmoid(x) = 1/(1 + exp(-x))`

![](sigmoid.png)


### Tanh function

- This is also a squisher function that squishes the inputs into a range of -1 to 1
- `tanh(x) = 1 - exp(-2x)/ 1 + exp(-2x)`

![](tanh.png)

### Exercises

1. Compute the derivative of the pReLU activation function.

* made a way to describe the function but the torch autograd is not able to work

`alpha = 0.1`
`y = find_max(X) + alpha * find_min(X)`

with torch.max the answer:

![](answer1.png)

2. Show that an MLP using only ReLU (or pReLU) constructs a continuous piecewise linear
function.

* I guess we need to construct a multi layer perceptron here. dontknow.

3. Show that tanh(x) + 1 = 2 sigmoid(2x).

* through plotting a graph we can show.

![](answer3.png)

4. Assume that we have a nonlinearity that applies to one minibatch at a time. What kinds of
problems do you expect this to cause?

* maybe this would create problems like each min batch would be squished(scaled) differently.

### Answers from the forums

Question 2:
I think it should be this:

H = Relu(XW^(1) + b^(2))
y = HW^(2) + b^(2)
More detail in page 131.
I think it is more easy to think like this:
Relu(x) constructs a continuous piecewise linear function for every x\in R. So, it do not depend on whatever x is providing that x is continuous in R. So, Relu(Relu(x)*W+b) for example is also constructs a continuous piecewise linear function.

Question 4:
I think the most different between an MLP apply nonlinearity and MLP not apply nonlinearity is the time and complexity. In fact, MLPs applying nonlinearity such as Sigmoid and tanh are very expensive to calculate and find the derivative for gradient descent. So, we need something faster and Relu is a good choice to address these problem (6.x sigmoid).

## Multi layer perceptron

- we will try and define a multi layer percceptron model

mlp prediction

but investigate y_hat size.

![](mlp_prediction.png)

### Exercises
1. Change the value of the hyperparameter num_hiddens and see how this hyperparameter influences your results. Determine the best value of this hyperparameter, keeping all others
constant.

![](changing_hyperparameters.png)


2. Try adding an additional hidden layer to see how it affects the results.

```python
# increasing number of hidden layers

W1 = nn.Parameter(torch.randn(num_inputs, 128) * 0.01,requires_grad=True)
b1 = nn.Parameter(torch.zeros(128),requires_grad=True)
W2 = nn.Parameter(torch.randn(128, 64) * 0.01, requires_grad=True)
b2 = nn.Parameter(torch.zeros(64), requires_grad=True)
W3 = nn.Parameter(torch.randn(64,num_outputs)*0.01, requires_grad=True)
b3 = nn.Parameter(torch.zeros(num_outputs),requires_grad=True)

def net(X):
    X=X.reshape(-1,num_inputs)
    out = relu(torch.matmul(X,W1) + b1)
    out = relu(torch.matmul(out,W2)+b2)
    return torch.matmul(out,W3) + b3
```

![](adding_hidden_layer.png)

3. How does changing the learning rate alter your results? Fixing the model architecture and
other hyperparameters (including number of epochs), what learning rate gives you the best
results?

* changes the rate of convergence

4. What is the best result you can get by optimizing over all the hyperparameters (learning rate,
number of epochs, number of hidden layers, number of hidden units per layer) jointly?

*  loss of 0.5

5. Describe why it is much more challenging to deal with multiple hyperparameters.

* combinatorial explosion because of more combination of hyperparameters

6. What is the smartest strategy you can think of for structuring a search over multiple hyperparameters?

* creating a matrices of all parameters and then optimally training over the combinationto find the result. some heuristic may be required.

## Concise Multi layer perceptron

![](concise_multilayerperceptron.png)

### Exercises
1. Try adding different numbers of hidden layers (you may also modify the learning rate). What
setting works best?

* it worked for me by just adding one more layer , and with SGD best earning rate was 0.1

2. Try out different activation functions. Which one works best?

* I tried ADAM but shd was the best activation.

3. Try different schemes for initializing the weights. What method works best?

* tried putting all linear layers to zero, but normal initialisation works best

![](concise_mlp_exercise.png)

## Model selection, Underfitting overfitting

- points to consider, memorising data is not good, space requirements make it infeasible, generalisationofafair cointoss if taken enough sample would always be 1/2. if its not then there is sampling bias.

- however model shouldnt be too generalised so as to catch onto spurious patterns

- more the number of tunable parameter more the tendency to overfit

- if more the range of value takenbu parameters more the case of overfitting

- the number of taining examples if less

- after model complexity ,model selection is important

- Validation dataset

- K fold validation -model is split into K subsets. And each time validation is done on the subset its not trained on. Validation error then is the average over K subsets.

- If validation and training error both are high, with a little gap between them it means we are undergitting. Our model is too simple. If training error is severly lower than validation error , it means we are overfitting.

- Dataset size -more data means morecomplex model.More data never hurts generally.

## Polynomial Regression

![](polynomial_regression.png)

### Demonstrating fit :

1. 3 variable normal fitting.

- errors I encountered, changing type of tensor using `torch.type`
- using `cross entropy loss` to do regresssion instead of `MSELoss`

![](three_variable_normal_fitting.png)


the graph:

![](normal_fit.png)

2. Creating a linear function. Underfitting.

![](underfit.png)

3. Overfitting a linear function

![](overfit.png)

### Exercises

1. Can you solve the polynomial regression problem exactly? Hint: use linear algebra.

* dont know how to do it. exactly.

2. Consider model selection for polynomials:
    1. Plot the training loss vs. model complexity (degree of the polynomial). What do you
    observe? What degree of polynomial do you need to reduce the training loss to 0?

    ![](loss_vs_degree.png)

    at about 5 degree it becomes zero.

    2. Plot the test loss in this case.
    
    3. Generate the same plot as a function of the amount of data.

* my graphs- 

![](loss_plots.png)

3. What happens if you drop the normalization (1/i!) of the polynomial features x
i
? Can you
fix this in some other way?

![](no_normalisation.png)

* Maybe we can put our own normalisation feature mechanism, though I am not sure.

4. Can you ever expect to see zero generalization error?

* only when the environment conditions are fully known.

## Weight decay

- Is a way for mitigating overfitting.
- removing features may turn out ot be costly to avoid overfitting.
- We need a way to deal with function gradients
- Weightdecay is l2regularisation, we add the norm as a penalising factor to the loss function
- in this case l2 means ||w||^2 multiplied bya regularisation parameter lambda/2
- if lambda=0 there is no regularisation, if >0 there is
- l1 regularisation is lasso regularisation, l2 regularisation is ridge regularisation
- l1 penalties lead to model creation from small number of parameters while l2 punished big weights in all parameters
- since we not only update the weight we also try to bring it closer to zero in l2 regularisation thats why it is known as weight decay.
- w <- (1-eta * lambda ) w - eta/batch_size * gradient of loss
here , gradient of loss being  sum over btach(x )* sum over batch(w.T * x + b - y)

### High dimensional linear regression

Using linear regression on weight decay we have implementation from scratch:

![](weight_decay_surprise.png)

However when we use the default weight decay using SGD we get better result

![](weight_decay_concise.png)

### Exercises

1. Experiment with the value of λ in the estimation problem in this section. Plot training and
test accuracy as a function of λ. What do you observe?

* as we increase wd it is making better train and test loss graph

![](weight_decay_loss.png)

2. Use a validation set to find the optimal value of λ. Is it really the optimal value? Does this
matter?

* the minimum loss is 0.02 at wd = 21-5 = 16. It is th eoptimal value for the validation set.

![](valid_wd.png)

3. What would the update equations look like if instead of ∥w∥
2 we used ∑
i
|wi
| as our penalty
of choice (L1 regularization)?

* It is strangely coming as linear

```python
# training for l1 norm
model = nn.Sequential(nn.Linear(num_inputs, 1))

for param in model.parameters():
    param.data.normal_()
    print(l1_norm(param.detach()))
    
loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.03)

valid_loss_array = []
for epoch in range(20):
    current_loss = 0
    current_number = 0
    for X, y in valid_dataloader:
        l = loss(model(X),y) - l1_norm_with_abs(model[0].weight)
        
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        current_loss += l.detach()
        current_number += len(y)
    
    valid_loss_array.append(current_loss/current_number)

plt.plot(range(20), valid_loss_array)
plt.grid(True)
plt.show()

```

![](l1_norm.png)

4. We know that ∥w∥
2 = w⊤w. Can you find a similar equation for matrices (see the Frobenius
norm in Section 2.3.10)?

* This is Frobenius norm: f(αx) = |α|f(x).  I dont understand.

5. Review the relationship between training error and generalization error. In addition to
weight decay, increased training, and the use of a model of suitable complexity, what other
ways can you think of to deal with overfitting?

* larger data size, dropouts

6. In Bayesian statistics we use the product of prior and likelihood to arrive at a posterior via
P(w | x) ∝ P(x | w)P(w). How can you identify P(w) with regularization?

* you ask good question but me no understand. how to get to p of w.

## Dropout

- Overfitting revisited - with overfitting, if you have more examples thatn features youll not overfit generally.
- bias variance tradeoff - chooseing between flexibility and generalisation, like linear models have higher bias they can representonly a small number of functions, however the variance is less.

- Neural network on th eother hand have a tendency to find associations between the features.

- in 1995, Christopher bishop showed that adding noise was going to add regularisation, this idea was adopted by Srivastava et al in 2014 for creating dropout

- standard dropout means zeroing someof the inputs at each layer. it was meant to break the coadaptation.

- originally by bishop ateach training iteration, headded agauusian normal noise, but in dropout, in dropout the activation is replaced by either 0 or h/ 1-p

- we generally dont apply dropout over testing data, unless we need to use it as a heuristic to measure how confident the model is.

### Creating a dropout based prediction model from scratch

![](dropout_from_scratch.png)

### Creating a concise model

![](dropout_concise.png)

### Exercises

1. What happens if you change the dropout probabilities for the first and second layers? In
particular, what happens if you switch the ones for both layers? Design an experiment to
answer these questions, describe your results quantitatively, and summarize the qualitative
takeaways.

* Nothing much changes based on the final accuracy to compare. 

![](dropout_ex1.png)

2. Increase the number of epochs and compare the results obtained when using dropout with
those when not using it.

* there is a significant difference in loss. Iam getting comparative results in both.

![](dropout_ex2.png)

3. What is the variance of the activations in each hidden layer when dropout is and is not applied? Draw a plot to show how this quantity evolves over time for both models.

* What is meant by variance here do we just apply torch.var over the net.linear weight layers? thisis the graph I came up with.

![](dropout_ex3.png)

4. Why is dropout not typically used at test time?

* Because we would like all the features weights to be useful and consequential to making the prediction.

5. Using the model in this section as an example, compare the effects of using dropout and
weight decay. What happens when dropout and weight decay are used at the same time?
Are the results additive? Are there diminished returns (or worse)? Do they cancel each other
out?

* The returns are diminished when using both, compared to using them individually.

![](dropout_ex5.png)

6. What happens if we apply dropout to the individual weights of the weight matrix rather than
the activations?

* it is still training. But with some slight oscillation in loss. This is the forward implementation
 ```python
  def forward(self, X):
        out = self.lin1(X.reshape(-1,784))
        out = dropout_layer(out, self.dropout1)
        out = self.relu(out)
        out = self.lin2(out)
        out = dropout_layer(out, self.dropout2)
        out = self.relu(out)
        out = self.lin3(out)

        return out
        
 ```

![](dropout_ex6.png)

7. Invent another technique for injecting random noise at each layer that is different from the
standard dropout technique. Can you develop a method that outperforms dropout on the
Fashion-MNIST dataset (for a fixed architecture)?

* I propose to add noise after each hidden layer.

```python
class net_epsilon(nn.Module):
    def __init__(self, num_inputs=784, num_outputs=10, num_hidden1=256, num_hidden2=256, epsilon_m=0.02, epsilon_d=0.02):
        super(net_epsilon, self).__init__()
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()
        self.epsilon_m = epsilon_m
        self.epsilon_d = epsilon_d
        self.num_inputs = num_inputs
    
    def forward(self, X):
        out = self.relu(self.lin1(X.reshape(-1,self.num_inputs)))
        out = out + torch.normal(self.epsilon_m, self.epsilon_d, (out.shape))
        out = self.relu(self.lin2(X.reshape(-1, self.num_inputs)))
        out = out + torch.normal(self.epsilon_m, self.epsilon_d, (out.shape))
        out = self.lin3(out)
```

![](dropout_ex7.png)


## Forward propagation, backpropogation and computational graphs

- so far we only cared about forward propagation preferring tohandle back proagationthrough grad function in pytorch.

### Forward propagation

- Forward propogation is nothing but multiplying th eweights with features putting it thorugh an activation function and then figuring out loss, comparing it with base truth.

### Computational graph with forward propagation

- computational graphshows the relationship of various operator and operands.

### Backpropagation

- Backpropagation refers to the method of calculating the gradient of neural network parameters.
- method traverses  in reverse order from output to input
- given Y = f(X) and Z = g(Y) then the algorithm calculates dZ/dX = dZ/dY * dY/DX
- more to come here

### Exercises

1. Assume that the inputs X to some scalar function f are n × m matrices. What is the dimensionality of the gradient of f with respect to X?

* Dimensionality reamains the same I tried be look at X.grad values.

![](grad_shape.png)

2. Add a bias to the hidden layer of the model described in this section (you do not need to
include bias in the regularization term).
1. Draw the corresponding computational graph.
2. Derive the forward and backward propagation equations.

* given in the pic. There is almost no change.

![](working_out_gradients.jpeg)

3. Compute the memory footprint for training and prediction in the model described in this
section.

* If we are to assume that X belongs to R^d then , you would need to store dl/do, dh/dz, X, lambda, W(1), W(2), h, X.grad so considering X size is d then 4d + 4. Based on gradients calculated above. Rest all can be derived.

4. Assume that you want to compute second derivatives. What happens to the computational
graph? How long do you expect the calculation to take?

* dont know how to come up with a solution to this

5. Assume that the computational graph is too large for your GPU.
    1. Can you partition it over more than one GPU?

    * maybe computational grapha can be devided but there needs to be tmiing pipeline else one part would wait for the other. Maybe different batchwise the multiple GPU can be recruited.

    2. What are the advantages and disadvantages over training on a smaller minibatch?

    * smaller minibatch takes more time and less memory and vice versa. Its a time space tradeoff.

My question now is what does X.grad give you what is that computation?

Conceptually, autograd records a graph recording all of the operations that created the data as you execute operations, giving you a directed acyclic graph whose leaves are the input tensors and roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule.

## Numerical stability and initialisation

- The choice of initialisaition and the choice ofnon linear function is especially important lest it leads us to exploding or vanishing gradients.

### Vanishing and exploding gradients

- since we havea depp neural network we would have have multiplelayers of of activation to deal with , if h(l) = f(h(l-1)) when we derive the derivative d/dl*(h(l)) it would be W(h) * h(l-1)-> similarly for rest soit eventually become a product of all the layer weights, if the weights are too small then it creates vanishing gradient problem if too large then exploding gradient.

- Sigmoid activation and problem of vanishing gradients.

![](sigmoid_exploding.png)

if the weights as=re too large or too small sigmoid vanishes.

- if too large then also goesto infinity.

![](exploding_gradients.png)

- breaking the symmetry.
with same initialisation it might happenthat both hidden network getactivated in the same way, thus making multiple layers redundant.

### Parameter initialisation 

-  Default initialisation - either the system will initialise on its own or using normal initialisation.

- Xavier initialisation -  we know that o = sumof(w*x)
now if we take mean to be zero and variance to gamma^2 then, E(o) = sum of (E(w) E(x)) since x =0 => E(o) 0

- similarly var (o) = E(o^2) - E(o)^2 = E(o^2) = E(w^2x^2) =  n * sigma^2 * gamma^2

if we take n * sigma^2  = 1 then, taking similar condition for outpt we have, 1/2* (nin + nout) ^2 * sigma^2 = 1 => sgma = dq_root((2/nin + nout)^2)

Note that the uniform distribution $U(-a, a)$ has variance $\frac{a^2}{3}$.
Plugging $\frac{a^2}{3}$ into our condition on $\sigma^2$
yields the suggestion to initialize according to

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$


### Exercises

1. Can you design other cases where a neural network might exhibit symmetry requiring breaking besides the permutation symmetry in an MLP's layers?

* loss and regularisation 

1. Can we initialize all weight parameters in linear regression or in softmax regression to the same value?

* We can try it but it may lead to symmetry condition

1. Look up analytic bounds on the eigenvalues of the product of two matrices. What does this tell you about ensuring that gradients are well conditioned?

* The largest eigenvalue of such a matrix (symmetric) is equal to the matrix norm. Say your two matrices are A and B.

∥AB∥≤∥A∥∥B∥=λ1,Aλ1,B

where λ1,A is the largest eigenvalue of A and λ1,B is the largest eigenvalue of B. So the largest eigenvalue of the product is upper-bounded by the product of the largest eigenvalues of the two matrices. For a proof of what I just asserted, see: Norm of a symmetric matrix equals spectral radius

In terms of the smallest, it looks like the product of the smallest two eigenvalues also gives you a lower bound on the smallest eigenvalue of the product. For a complete reference on how the eigenvalues are related, see: https://mathoverflow.net/questions/106191/eigenvalues-of-product-of-two-symmetric-matrices

* Condition number (l) = max value of eigen value divided by min value of eigen value.
If l >> 1 then it we say it is ill-conditioned
If l approx = 1 the well-conditioned.

1. If we know that some terms diverge, can we fix this after the fact? Look at the paper on layerwise adaptive rate scaling  for inspiration :cite:`You.Gitman.Ginsburg.2017`.

* LARS uses seperate learning rate for each layer.







