# StackedNet - Lightweight greedy layer-wsie training 

The "StackedNet" class implemented Staked Sparse Autoencoders, on the top
of Theano so as to use GPU acceleration.

- Motivation
    Most of the open access libraries do not support greedy layer-wise
    initialization, developers and researches are suffer from implement it.
    For this reason, I release this lightweight code which is easy to
    modify.

- start_greedy_layer_training (unsupervised)
    is used to initialize the weights in the layer-wise manner, using Sparse
    Autoencoder, sigmoid and mean square error by default. This class is
    easy to modify to any expected behaviour.

- start_fine_tune_training (supervised)
    is used to train the whole network after greedy layer training, using
    softmax output and cross-entropy by default, without any dropout and
    regularization.
    However, this example will save all parameters' value in the end, so the
    author suggests you to design your own fine-tune behaviour if you want
    to use dropout or dropconnect.

- This code available for python 2 and python 3

- author:
    Hao Dong
    Department of Computing & Data Science Institute
    Imperial College London



"StackedNet" 类，基于Theano的GPU加速，实现了栈式稀疏自编码器

- 目的
	目前开源的深度学习库基本都不提供逐层训练的功能，开发者和研究者需要花很多精力来自己
	开发这些功能。因此，发布这个非常容易做二次修改的轻量级的代码，可以帮助大家提高效率。

- start_greedy_layer_training (unsupervised)
	该方法用于隐层的逐层预训练，以初始化weights。每个隐层默认为稀疏自编码器，使用sig-
	moid函数和mean square error作为损失函数。用户可以自行修改损失函数，以实现自己想
	要的效果。

- start_fine_tune_training (supervised)
	该方法用于微调逐层训练后的整个神经网络，使用softmax作为输出层，cross-entropy作为
	损失函数，没用使用任何的dropout和regularization。
	然而，该代码例子会保存所有weights和bias的值，若用户想要设计自己的微调方法，作者建
	议自己重写微调算法，或者用其他开源库来微调之。

- 该代码支持 python 2 和 python 3


- 作者:
    董豪 
    计算机系 及 数据科学研究所  
    伦敦帝国理工 

