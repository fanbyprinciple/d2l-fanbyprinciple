## Alexnet

Trained alex net.

![](alex_params.png)

Memory footprint.

![](alex_trained.png)

## VGG 

![](vgg_params.png)

Need to revisit softmax, negative log liklihood (page 126), nn.CrossEntropyLoss(136), SGD

it did not train at 0.05 or 0.1.

Tried again it is training at 0.05.

Another problem that I came accross wass efficient memory management. The tensors I create stay in the GPU. how to delete them after use.

![](VGG_result.png)

## Network in network

![](nin_training.png)

Not exactly training properly.


