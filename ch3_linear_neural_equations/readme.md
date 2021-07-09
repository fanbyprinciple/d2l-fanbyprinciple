# Linear Neural equations

- Learning simple neural network thorugh linear regression.

## stuff to refer

- log liklihood loss
- basic loss estimation and parameter updation
-  

## Linear regression

- Regression refers to a set of methods for modeling the relationship between one or more independent variables and a dependent variable. In the natural sciences and social sciences, the purpose
of regression is most often to characterize the relationship between the inputs and outputs. Machine learning, on the other hand, is most often concerned with prediction

### Basic elements of linear regression

- prices of houses based on their area , x1 and x2 as inputs and y as output

- linear model - `price = w_area * x1 + w_sq_ft * x2 + b`
 Strictly speaking, (3.1.1) is an affine transformation of input features, which is characterized by a linear transformation of features via weighted sum, combined with a translation via the added bias.

- y_ = W.T * X + b

- Loss function - 1/2 (y - y_)^2 - 
    1/2 is just convenient when we take derivative
    over the entire model we have , 
    1/n (sum(1/2(y-y_)^2))
    = 1/n (sum(1/2(Xw + b - Y)^2))

- we want to find w and b such that we minimise the loss, argmin(L(w,b))

- Analytic solution - 
    1. Linear regression happens to be an unusually simple optimization problem. Unlike most other
    models that we will encounter in this book, linear regression can be solved analytically by applying
    a simple formula. 
    
    2. To start, we can subsume the bias b into the parameter w by appending a column
    to the design matrix consisting of all ones. 
    
    3. Then our prediction problem is to minimize ∥y−Xw∥^2
    
    4. There is just one critical point on the loss surface and it corresponds to the minimum of the loss
    over the entire domain. Taking the derivative of the loss with respect to w and setting it equal to
    zero yields the analytic (closed-form) solution:

    w∗ = (X.⊤ * X)^−1 * X.⊤ * y

- minibatch stochastic gradient descent - 
    1. The key technique for optimizing nearly any deep learning model, and which we will call upon
    throughout this book, consists of iteratively reducing the error by updating the parameters in the
    direction that incrementally lowers the loss function. This algorithm is called gradient descent.

    2. The most naive application of gradient descent consists of taking the derivative of the loss function, which is an average of the losses computed on every single example in the dataset. In practice, this can be extremely slow: we must pass over the entire dataset before making a single
    update. Thus, we will often settle for sampling a random minibatch of examples every time we
    need to compute the update, a variant called minibatch stochastic gradient descent.

    3. In each iteration, we first randomly sample a minibatch B consisting of a fixed number of training
    examples. We then compute the derivative (gradient) of the average loss on the minibatch with
    regard to the model parameters. Finally, we multiply the gradient by a predetermined positive
    value η and subtract the resulting term from the current parameter values.

    4. In short,
    
    (i) we initialize the values of the model parameters, typically at random;
    (ii) we iteratively sample random minibatches from the data,
    updating the parameters in the direction of the negative gradient. For quadratic losses and affine
    transformations, we can write this out explicitly as follows:

    w_updated = w - (learning_rate/mini_batch_size) * sum_over_minibatch((X.w+b - Y)* X)
    b_updated = b - (learning_rate/mini_batch_size) * sum_over_minibatch((X.w + b - Y))

    5. These parameters that are tunable but not updated in the training loop are called hyperparameters

    6. Making predictions -  w_updated.T * x + b_updated

### Vectorization for speed

- drastic increase in efficiency

### Normal distribution and squared loss

- normal distribution can be visualised 
- 1/ root(2 * sigma ^2 * pi) * exp(-1/(2 * sigma ^ 2) * (x - mu)^2)

![](normal_distribution.png)

### log likelihood loss

- One way to motivate linear regression with the mean squared error loss function (or simply
squared loss) is to formally assume that observations arise from noisy observations, where the
noise is normally distributed as follows:
y = w.T *x + b + epsilon which is the normal noise

-predicting liklihood that we can see y when given a y is  -

- 1/root(2 * sigma^2 * pi) * exp(-0.5/ sigma^2 * (y- w.T * X - b)^ 2)

![](likelihood_of_x.png)

- sigma value doesnt effect log liklihood loss

## Linear regression to deep neural networks

- gives the biological model of neural network

### Exercises

1. Assume that we have some data
∑x1, . . . , xn ∈ R. Our goal is to find a constant b such that

(xi − b)2
is minimized.

    1. Find a analytic solution for the optimal value of b.

    xi = b

    2. How does this problem and its solution relate to the normal distribution?


2. Derive the analytic solution to the optimization problem for linear regression with squared
error. To keep things simple, you can omit the bias b from the problem (we can do this in
principled fashion by adding one column to X consisting of all ones).

1. Write out the optimization problem in matrix and vector notation (treat all the data as
a single matrix, and all the target values as a single vector).
2. Compute the gradient of the loss with respect to w.
3. Find the analytic solution by setting the gradient equal to zero and solving the matrix
equation.
4. When might this be better than using stochastic gradient descent? When might this
method break?
3. Assume that the noise model governing the additive noise ϵ is the exponential distribution.
That is, p(ϵ) = 1
2
exp(−|ϵ|).
1. Write out the negative log-likelihood of the data under the model − log P(y | X).
2. Can you find a closed form solution?
3. Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint: what happens near the stationary point as we keep on updating
the parameters)? Can you fix this?

## Creating linear neural networks from scratch

![](self_made_dataset.png)

