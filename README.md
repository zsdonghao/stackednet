# StackedNet

The "StackedNet" class implemented Staked Sparse Autoencoders, on the top
of Theano so as to use GPU acceleration.

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

- Motivation
    Most of the open access libraries do not support greedy layer-wise
    initialization, developers and researches are suffer from implement it.
    For this reason, I release this lightweight code which is easy to
    modify.

- author:
    Hao Dong
    Department of Computing & Data Science Institute
    Imperial College London
