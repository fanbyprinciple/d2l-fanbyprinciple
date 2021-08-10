# Ch5 Deep Learning computation

- deep learning is not only helped by sophisticated hardware but also through progressindeep learning libraries.

## Layers and Block

- In softmax regression a single layer was the model, however in multi layer perceptron we had many such layers makinga model.
- When it comes to complex models we abstract it into a group of layers like convolutional neural network or linear networks.

- nn.Sequential is away to create sub module in pytorch
- net(X)is a shorthand for net.__call__(X)

- If we are to create our own module we can do it through getting the input function, generating output function as returning a value, andthen catering to initalisation, back propogationshould be handled on its own

- We can also implement Sequential function of the sequential block- its main functionality is todaisy chain blocks together.

- creating a fixed mlp class

### Exercises
1. What kinds of problems will occur if you change MySequential to store blocks in a Python
list?

* Mysequential implementation would be different.

2. Implement a block that takes two blocks as an argument, say net1 and net2 and returns
the concatenated output of both networks in the forward propagation. This is also called a
parallel block.

* 
```python
class parallel_mlp(nn.Module):
    def __init__(self, block1, block2):
        super().__init__()
        self.block1 = block1
        self.block2 = block2

    def forward(self, X):
        first = self.block1(X)
        second = self.block2(X)
        print(first, second)
        return torch.cat((first, second))
```
![](ex51_parallel.png)

3. Assume that you want to concatenate multiple instances of the same network. Implement
a factory function that generates multiple instances of the same block and build a larger
network from it.

![](ex51_hydra.png)

