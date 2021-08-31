# Ch7 Mordern Convolutional Networks

* we would be using multiple architectures and implementing it

* reason for the growth - 1. good dataholding capacity 2. good gpu

## Alexnet

- won the image recognition contest in 2012 with a phenomenal margin.

- Here, we use a larger 11 x 11 window to capture objects. At the same
time, we use a stride of 4 to greatly reduce the height and width of the
output. Here, the number of output channels is much larger than that in
LeNet

however alexnet takesa longtime and also is not training,
![](alexnet_wrong.png)

Due to memory error goota try on kaggle to train: 

https://www.kaggle.com/fanbyprinciple/implementing-cnns-in-the-name-of-fashion/edit

However facing cuda outof memrot error for batch size 128 and 224,
![](kaggle_128.png)

and batch size 64 :
![](kaggle_64.png)

Intrestingly batch size of 64 uses more memory.

Even google colab says no:
![](google_colab_128.png)

https://colab.research.google.com/drive/1a7BYx_BboxomAJ4vSs8o3UoDy4ceC0e_#scrollTo=uymUYJeKxv2E


### Exercises
1. Try increasing the number of epochs. Compared with LeNet, how are the results different? Why?

Finally was able to "train" it but it looks more like a heartbeat than anything

![](alexnet_heartbeat.png)


* The result is a very noisy function, it is like its not getting trained at all.

2. AlexNet may be too complex for the Fashion-MNIST dataset.
    1. Try simplifying the model to make the training faster, while ensuring that the accuracy
    does not drop significantly.

    * done but accuracy has still not improved.

    2. Design a better model that works directly on 28 × 28 images.

    * done. 

3. Modify the batch size, and observe the changes in accuracy and GPU memory.

* more batch size more consumption

4. Analyze computational performance of AlexNet.

    * any idea how to do it ?

    1. What is the dominant part for the memory footprint of AlexNet?
        * Linear network and 3 cnns
    2. What is the dominant part for computation in AlexNet?
        * the neuralnetwork 3 cnns
    3. How about memory bandwidth when computing the results?
        * more than 100 mb
5. Apply dropout and ReLU to LeNet-5. Does it improve? How about preprocessing?

   ```python
    
    lenet_5 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
   ```

    its not training well as well, ![](lenet_heartbeat.png)

Trying to train a new network to verify my accuracy

https://www.kaggle.com/fanbyprinciple/cifar10-with-pytorch/edit

This is atleast more like it
![](pytorch_cifar.png)

Based on that I created another notebook for a smaller alexnet model

https://www.kaggle.com/fanbyprinciple/alex-net-small/edit

![](alex_net_small.png)

## VGG network








