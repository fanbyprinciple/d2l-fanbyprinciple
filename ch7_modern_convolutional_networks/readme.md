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





