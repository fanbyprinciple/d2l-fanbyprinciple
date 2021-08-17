# Ch6 Convolutional neural network

- problem of using the same approach of training a network as we did with tabular data, makes it infeasible to do it.
- translational invariance - our network should respond similarly in same patch
- locality principle - earliestlayers should focus on local regions

### contraining the mlp

- `H(i,j) = Bias(i,j) + sumover(a,b) * W(i,j,a,b) * X(i+a, j+b)`
where a and both go over the entire dataset.

### Translational invariance

- it simply means that a shift in value of X should lead to a shift invalue of hidden representation.

`H(i,j) = Bias + sumover(a,b) * V(i,j) * X(i+a, j +b)`

### Locality principle

- we should not look far from location (i, j) to glean important information about H(i,j)

`outside some delta |a| > delta or |b| > delta , V(i,j) = 0`
`H(i,j) = Bias _ sumover(-delta, delta) (a,b) V(a,b) * X(i+a, j+b)`

- These two principles help us narrow down the parameters for our hypothesis.

## Convolutions

- Lets briefly look why the operations are known as convolutions.

how aconvolution is defined,

`(f * g)(x) = integration(f(z) * g(x-z) * dz)`

for dicrete functions f and g, we have

`(f * g)(i) =  sumover(a) (f(a))* f(i-a))`

for two dimensional tensors

`(f * g)(a,b) = sumover(a)sumover(b) * f(a,b) * g(i-a, j-b)`

- this is similar to earlier equation thatwe cameup with for convolution.

### Channels

- images are not usually 2dimenisonal but 3 dimensional

-They are also sometimes called feature
maps, as each provides a spatialized set of learned features to the subsequent layer. Intuitively,
you might imagine that at lower layers that are closer to inputs, some channels could become
specialized to recognize edges while others could recognize textures.

![](channel_convolutions.png)

### Exercises

1. Assume that the size of the convolution kernel is ∆ = 0. Show that in this case the convolution kernel implements an MLP independently for each set of channels.

* when delta is zero in equation 6.1.7, only the sum over channel remains, which is as good as indiviually summing for each channels.

2. Why might translation invariance not be a good idea after all?

* if image are not translation invariant then that would be a problem, in case of video?

3. What problems must we deal with when deciding how to treat hidden representations corresponding to pixel locations at the boundary of an image?

* at boundary of image the offset would not be available hence padding would be required.

4. Describe an analogous convolutional layer for audio.

* for audio since we would need frequency and time to be two input layer, we need to make tehm into two features and then proceed to use the same formula for convolutional neural network for 2 parameters.

5. Do you think that convolutional layers might also be applicable for text data? Why or why
not?

* text data features would be equal to the vocabulary and they are not location dependent, even if we do a visual conversion of text data.

6. Prove that f ∗ g = g ∗ f

`(f * g)(a,b) = sumover(a)sumover(b) * f(a,b) * g(i-a, j-b)`
`(g * f)(a,b) = sumover(a)sumover(b) * g(a,b) * f(i-a, j-b) = (g * f)(c,d) = sumover(c)sumover(d) * g(i-c,j-d) * f(c, d)`
`for some c,d`

* Here is something I've sometimes wondered about. If f,g are both nonnegative proving commutativity of convolution can be done without a tedious change of variable.

* Indeed, let X be a random variable with density f and let Y be a random variable with density g. Its easy to see that f convolved with g is the density of X+Y (or in your case X+Y mod 2π). By commutativity of addition, the density of X+Y is the same as the density of Y+X and we are done!

# Convolution for images in practice

- convolution is a misnomer since what we use is cross correlation.

![](cross_correlation.png)

formula fo size after convolution - 

nh and nw are for hidden layer
kh and kw are for kernel,

(nh − kh + 1) × (nw − kw + 1)


