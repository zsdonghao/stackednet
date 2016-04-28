from scipy.io import savemat, loadmat
import numpy as np
import sys
rng = np.random
import os
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


def load_dataset():
    import gzip
    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28*28)
        return data / np.float32(256)
    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    data_dir = os.getcwd() + '/gvm/'

    X_train = load_mnist_images(data_dir+'train-images-idx3-ubyte.gz') # DH modify
    y_train = load_mnist_labels(data_dir+'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(data_dir+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(data_dir+'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_val = []
    y_val = []
    # X_train, X_val = X_train[:-10000], X_train[-10000:]
    # y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluation(y_test, y_predict, n_classes):
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    c_mat = confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)])
    f1    = f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)])
    f1_macro = f1_score(y_test, y_predict, average='macro')
    acc   = accuracy_score(y_test, y_predict)
    print('confusion matrix: \n',c_mat)
    print('f1-score:',f1)
    print('f1-score(macro):',f1_macro)   # same output with > f1_score(y_true, y_pred, average='macro')
    print('accuracy-score:', acc)
    return c_mat, f1, acc, f1_macro

class HiddenLayer(object):
    def __init__(
        self,
        inputs=None,
        n_units_previous=100,
        n_units=100,
        name='0'
    ):
        self.inputs = inputs
        self.n_units_previous = n_units_previous
        self.n_units = n_units
        self.name = name
        rng = np.random
        self.W = theano.shared(
                    np.asarray(rng.randn(n_units_previous, n_units), dtype=np.float32),
                    name="W"+name)
        self.b = theano.shared(
                    np.zeros(shape=(1, n_units), dtype=np.float32)[0],
                    name="b"+name)
        self.a = T.nnet.sigmoid(T.dot(inputs, self.W) + self.b)



class StackedNet(object):
    """ Stacked Auto-Encoder class (with denoising and exploding)

    A denoising, exploding autoencoder tries to reconstruct the inputs

    .. math::
        Assume the autoencoder have 2 hidden layers
        784 -> 500 -> 500 -> 10
        a[0] = sigmoid( dot(x, W1) + b1 )
        a[1] = sigmoid( dot(a[0], W2) + b2 )
        h = softmax( dot(a[1], W_out) + b_out)
        y = argmax(h)
        ...
        Greedy layer-wise training ---
        cost = mean-squared-error + beta * sparsity + L2 + l2_decoder + pi * exploding
        Fine-turning ---
        cost = cross-entropy
    .. author:
        Hao Dong, Department of Computing, Imperial College London
        2016
    """
    def __init__(
        self,
        # numpy_rng,
        # theano_rng=None,
        inputs=None,
        targets=None,
        n_visible=784,
        n_units_hidden=[400, 400],
        n_classes=10,
        W_exist=[],
        b_exist=[],
        n_epochs=[1000, 1000, 300],
        batch_size=[100, 100, 100],
        learning_rate = [0.0001, 0.0001, 0.001],
        update='adam',
        dp=[0, 0, 0],
        beta=[4, 4],
        p_sparsity=[0.15, 0.15],
        l2_lambda = [0.004, 0.004, 0],
        l2_decoder_lambda=[0., 0.],
        pi=[4, 4],
        pruning=[0.02, 0],
        uints_le_previous=True,
        print_freq=10,
    ):
        """
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input:

        :type target: theano.tensor.TensorType
        :param target: target prediction/ label

        :type n_visible: int
        :param n_visible: number of visible units/ input units

        :type n_units_hidden: list of int               len() = len(n_hidden_layer)
        :param n_units_hidden:  initial number of hidden units of each hidden layer
                                will be changed if neurons are 'broken'

        :type n_classes:  int
        :param n_classes:  number of classes, i.e. number of softmax output

        :type W_exist: list of numpy.array
        :param W_exist: the existing weights of autoencoder layers provided by user
                        if no, leave it empty

        :type b_exist: list of numpy.array
        :param b_exist: the existing bias of autoencoder layers provided by user
                        if no, leave it empty

        :type n_epochs: list of int                     len() == len(n_hidden_layer) + 1
        :param n_epochs: the training epochs of each hidden layers and output softmax
                         last int is for fine-turn

        :type batch_size: int                           len() == len(n_hidden_layer) + 1
        :param batch_size: mini-batch number, last int is for softmax

        :type learning_rate: list of float              len() == len(n_hidden_layer) + 1
        :param learning_rate: last float is for softmax fine-turning

        :type update: string
        :param update: 'adam': adaptive learning rate   'fix': fixed learning rate

        :type dp: list of float                         len() == len(n_hidden_layer) + 1
        :param dp: dropconnect probability of each layers (probability of forcing values to 0)
                   1st float  : input layer to 1st hidden layer-wise (if >0, it is denoising AE)
                   last float : last hidden layer to output layer

        :type beta: list of float                       len() == len(n_hidden_layer)
        :param beta: scale value of sparsity_penalty

        :type p_sparsity: list of float                 len() == len(n_hidden_layer)
        :param p_sparsity: target sparsity ratio

        :type l2_lambda: list of float                  len() == len(n_hidden_layer) + 1
        :param l2_lambda: when using vanila sparse AE
                          regularization for encoder's weights and decoder's weights during greedy layer-wise training
                          last float is for fine-turn training

        :type l2_decoder_lambda: list of float          len() == len(n_hidden_layer)
        :param l2_decoder_lambda: when using exploding AE
                          regularization for weights of decoder during greedy layer-wise training

        :type pi: list of float                         len() == len(n_hidden_layer)
        :param pi: scale value of exploding penalty

        :type pruning: list of float                    len() == len(n_hidden_layer)
        :param pruning: threshold value for pruning broken neuron
                        if the std of the weights on a neuron < pruning, remove the neuron

        :type uints_le_previous: boolean
        :param uints_le_previous: if True, the number of neuron of a hidden layer
                                  is forced to less than or equal to its previous layer,
                                  the neuron less than previous layer, if only if explode
                                  happen in the current layer.

        :type print_freq: int
        :param print_freq: the frequency of printing state of epochs during training

        """
        assert len(n_epochs) == len(batch_size) == len(dp) == len(l2_lambda) == len(n_units_hidden) + 1, " argument(s) error "
        assert len(beta) == len(p_sparsity) == len(l2_decoder_lambda) == len(pi) == len(pruning) == len(n_units_hidden), " argument(s) error "

        print("\nInitializing stacked denoising, exploding autoencoder ...")
        # self.numpy_rng = numpy_rng
        self.inputs = inputs
        self.targets = targets
        self.n_visible = n_visible
        self.n_hidden_layer = len(n_units_hidden)
        self.n_units_hidden = n_units_hidden
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.update = update
        self.dp = dp
        self.beta = beta
        self.p_sparsity = p_sparsity
        self.l2_lambda = l2_lambda
        self.l2_decoder_lambda = l2_decoder_lambda
        self.pi = pi
        self.pruning = pruning
        self.uints_le_previous = uints_le_previous
        self.print_freq = print_freq

        self.mytype = np.float32    # theano GPU only work with float32

        print("  number of visible units: %d" % self.n_visible)
        print("  number of hidden layers (AE layers): %d" % self.n_hidden_layer)
        print("  number of hidden units of each hidden layer (init): ", *self.n_units_hidden)

        # self.W = [] # list of tensor Sharedvariable
        # self.b = [] # list of tensor Sharedvariable
        self.layers = {} # list of layers
        self.layers[0] = self.inputs # the first layer is input layer
        # self.layers[0] = HiddenLayer(
        #                         inputs=self.inputs,
        #                         n_units_previous=0,
        #                         n_units=self.n_visible,
        #                         name='0'
        #                     )
        # self.layers[0].a = self.inputs  # 之后在改成这个形式吧

    def start_greedy_layer_training(self, X_train, start=1):
        """ greedy layer-wise training (self-taught, unsupervised)
        :type start: int
        :param int: start Greedy layer-wise training from which hidden AE layer
                    if start==1, start from 1st AE
        """
        rng = np.random
        print("\nStart Greedy layer-wise training AE(s)")
        for i in range(start-1, self.n_hidden_layer):
            print("Start Training AE %d" % (i+1) )
            # i = [0, 1]
            if i > 0:
                print("    pruning broken neurons of previous layer: %f" % self.pruning[i-1])
                # remove bad neurons of previous layer
                W_val, b_val = self.remove_bad_feature(self.layers[i].W.get_value(), self.layers[i].b.get_value(), threshold=self.pruning[i-1])
                self.layers[i].W.set_value(np.asarray(W_val, dtype=self.mytype))
                self.layers[i].b.set_value(np.asarray(b_val, dtype=self.mytype))
                del W_val, b_val
                # reset the n_units_hidden of previous layer
                self.n_units_hidden[i-1] = self.layers[i].W.get_value().shape[1]
                self.layers[i].n_units = self.n_units_hidden[i-1]

                if self.uints_le_previous:
                    print("    user set num of neuron AE %d (less than or equal to) AE %d" % (i+1, i))
                    if self.n_units_hidden[i] > self.n_units_hidden[i-1]:
                        self.n_units_hidden[i] = self.n_units_hidden[i-1]

            if i == 0:
                self.layers[i+1] = HiddenLayer(     # 1st hidden layer
                                        inputs=self.inputs,
                                        n_units_previous=self.n_visible,
                                        n_units=self.n_units_hidden[i],
                                        name=str(i+1),
                                        )
            else:
                self.layers[i+1] = HiddenLayer(
                                        inputs=self.layers[i].a,
                                        n_units_previous=self.n_units_hidden[i-1],
                                        n_units=self.n_units_hidden[i],
                                        name=str(i+1),
                                            )


            W_decoder = theano.shared(
                            np.asarray(rng.randn(self.layers[i+1].n_units, self.layers[i+1].n_units_previous),
                            dtype=self.mytype), name="W_decoder")
            b_decoder = theano.shared(
                            np.zeros(shape=(1,self.layers[i+1].n_units_previous),
                            dtype=self.mytype)[0], name="b_decoder")
            h_decoder = T.nnet.sigmoid(T.dot(self.layers[i+1].a, W_decoder) + b_decoder)

            for key in self.layers:
                if key != 0: # key==0时，layer 是 self.inputs
                    print("   Info of hidden layer %d" % key)
                    print('   W',self.layers[key].W.get_value().shape)
                    attrs = vars(self.layers[key])
                    print('   , '.join("%s: %s\n" % item for item in attrs.items()))
            train_params = [self.layers[i+1].W, self.layers[i+1].b, W_decoder, b_decoder]
            # print(W_decoder.get_value().shape, b_decoder.get_value().shape)
            print("  train:",train_params)
            print("  batch_size:%d n_epochs:%d learning_rate:%f" % (self.batch_size[i], self.n_epochs[i], self.learning_rate[i]) )
            print("  use mse, l2_lambda:%f L2_W_decoder:%f beta:%f p_sparsity:%f pi:%f" %
                    (self.l2_lambda[i], self.l2_decoder_lambda[i], self.beta[i], self.p_sparsity[i], self.pi[i]) )

            if i==0:
                ce = T.nnet.categorical_crossentropy(h_decoder, self.inputs).mean()
                mse = ((h_decoder - self.inputs) ** 2).sum(axis=1).mean()
            else:
                ce = T.nnet.categorical_crossentropy(h_decoder, self.layers[i].a).mean()
                mse = ((h_decoder - self.layers[i].a) ** 2).sum(axis=1).mean()
            L2 = self.l2_lambda[i] * (  (self.layers[i+1].W ** 2).sum() + (W_decoder ** 2).sum()  )
            L2_W_decoder = self.l2_decoder_lambda[i] * (W_decoder ** 2).sum()
            w_col_sparsity = T.sqrt(T.sum(self.layers[i+1].W**2, axis=0)).mean()

            p_hat = T.mean( self.layers[i+1].a, axis=0 )
            sparsity_penalty = T.sum( self.p_sparsity[i] * T.log(self.p_sparsity[i]/ p_hat) \
                                    + (1- self.p_sparsity[i])* T.log((1- self.p_sparsity[i])/(1- p_hat)) )
            cost = mse
            if self.beta[i] > 0:
                cost += self.beta[i] * sparsity_penalty
            if self.l2_lambda[i] > 0:
                cost += L2
            if self.l2_decoder_lambda[i] > 0:
                cost += L2_W_decoder
            if self.pi[i] > 0:
                cost += self.pi[i] * w_col_sparsity


            if self.update == 'adam':
                updates = self.adam(cost, train_params, learning_rate=self.learning_rate[i])
            elif self.update == 'fix':
                updates = self.fixupdate(cost, train_params, learning_rate=self.learning_rate[i])
            else:
                raise Exception("Unknow update method")

            # # X: Prof in the theano.function used to check GPU status via profmode.print_summary()
            # from theano import ProfileMode  # http://deeplearning.net/software/theano/tutorial/modes.html#using-profilemode
            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            # profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
            train_fn = theano.function(
                      inputs=[self.inputs],
                      outputs=[cost],
                      updates=updates,
                    #   mode=profmode   # X: check GPU
                       )
            check_fn = theano.function(
                       inputs=[self.inputs],
                       outputs=[mse, ce, self.beta[i] * sparsity_penalty, L2, L2_W_decoder, self.pi[i] * w_col_sparsity]
                       )
            import time
            epoch = 0
            test_acc_val = 0
            for j in range(self.n_epochs[i]):
                start_time = time.time()
                train_err = 0
                train_batches = 0
                for (X_train_a, _) in self.iterate_minibatches(X_train, X_train, self.batch_size[i], shuffle=True):
                    train_loss_val = train_fn(X_train_a)[0]
                    train_err += train_loss_val
                    train_batches += 1
                # _ , train_loss_val= train_fn(X_train, X_train)

                if epoch + 1 == 1 or (epoch + 1) % self.print_freq == 0:
                    # test_loss_val = test_fn(X_test)
                    print("Epoch {} of {} took {:.3f}s".format(
                                epoch + 1, self.n_epochs[i], time.time() - start_time))
                    print("  training loss: %.10f" % float(train_err / train_batches))
                    # print("  test loss:     %5.5f" % test_loss_val[0])
                    print("  mse: ce: sparse: L2: L2_W_decoder: w_col:")
                    print(" ", *check_fn(X_train[0:50000:5]))  # load less data to avoid out of memory on GPU
                # if i == 0 and epoch == self.n_epochs[i]-1:
                    self.visualize_assquare_W(self.layers[i+1].W.get_value(), self.n_units_hidden[i], second=0, saveable=True, idx='w1_'+str(epoch+1) )
                # if epoch == 10-1:     # X: Check GPU
                #     # Spent 11.119s(59.336%) in cpu Op, 7.620s(40.664%) in gpu Op and 0.000s(0.000%) transfert Op
                #     profmode.print_summary() # GPU % CPU % usage and how to speed up
                #     exit()
                epoch = epoch + 1


    def start_fine_turn_training(self, X_train, y_train, X_val, y_val):
        # fine-turn (supervised)
        print("\nStart Fine-turning AE(s) using softmax output")
        print("    pruning broken neurons of last hidden layer: %f" % self.pruning[-1])
        W_val, b_val = self.remove_bad_feature(self.layers[self.n_hidden_layer].W.get_value(), self.layers[self.n_hidden_layer].b.get_value(), threshold=self.pruning[-1])
        self.layers[self.n_hidden_layer].W.set_value(W_val)
        self.layers[self.n_hidden_layer].b.set_value(b_val)
        # reset the n_units_hidden of last hidden layer
        self.n_units_hidden[-1] = W_val.shape[1]
        self.layers[self.n_hidden_layer].n_units = self.n_units_hidden[-1]
        del W_val, b_val

        self.layers[self.n_hidden_layer+1] = HiddenLayer(
                                inputs=self.layers[self.n_hidden_layer].a,
                                n_units_previous=self.n_units_hidden[-1],
                                n_units=self.n_classes,
                                name='out',
                                )

        self.layers[self.n_hidden_layer+1].a = T.nnet.softmax(
                                    T.dot(self.layers[self.n_hidden_layer].a, self.layers[self.n_hidden_layer+1].W) + self.layers[self.n_hidden_layer+1].b
                                    )
        self.y_pred = T.argmax(self.layers[self.n_hidden_layer+1].a, axis=1)

        for key in self.layers:
            if key != 0: # key==0时，layer 是 self.inputs
                print("   Info of Layer %d" % key)
                print("   W",self.layers[key].W.get_value().shape)
                attrs = vars(self.layers[key])
                print(',    '.join("%s: %s\n" % item for item in attrs.items()))

        train_params = []
        for key in self.layers:
            if key != 0:
                train_params.extend([self.layers[key].W, self.layers[key].b])
        print("  train:",train_params)

        print("  batch_size:%d n_epochs:%d learning_rate:%f" % (self.batch_size[-1], self.n_epochs[-1], self.learning_rate[-1]) )
        ce = T.nnet.categorical_crossentropy(
                    self.layers[self.n_hidden_layer+1].a, T.extra_ops.to_one_hot(self.targets, self.n_classes, dtype='int32')
                    ).mean()
        cost = ce
        if self.update == 'adam':
            updates = self.adam(cost, train_params, learning_rate=self.learning_rate[-1])
        elif self.update == 'fix':
            updates = self.fixupdate(cost, train_params, learning_rate=self.learning_rate[-1])
        else:
            raise Exception("Unknow update method")

        train_fn = theano.function(
                  inputs=[self.inputs, self.targets],
                  outputs=[cost],
                  updates=updates,
                #   mode=profmode
                #  mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) # NaN debug
                   )
        test_fn = theano.function(
                  inputs=[self.inputs, self.targets],
                  outputs=[cost]
                  )
        acc = T.mean(T.eq(self.y_pred, self.targets), dtype=theano.config.floatX)
        check_acc = theano.function(inputs=[self.inputs, self.targets], outputs=[acc, self.y_pred])

        import time
        epoch = 0
        val_acc_max = 0
        for j in range(self.n_epochs[-1]):
            start_time = time.time()
            train_err = 0
            train_batches = 0
            for (X_train_a, y_train_a) in self.iterate_minibatches(X_train, y_train, self.batch_size[-1], shuffle=True):
                train_loss_val = train_fn(X_train_a, y_train_a)[0]
                train_err += train_loss_val
                train_batches += 1
            # profmode.print_summary()
            # exit()

            # _ , train_loss_val= train_fn(X_train, X_train)
            val_acc = check_acc(X_val, y_val)[0]
            if val_acc_max < val_acc:
                val_acc_max = val_acc
            # if epoch + 1 == 1 or (epoch + 1) % self.print_freq == 0:
            val_loss = test_fn(X_val, y_val)
            print("Epoch {} of {} took {:.3f}s".format(
                        epoch + 1, self.n_epochs[-1], time.time() - start_time))
            print("  training loss:   %.10f" % float(train_err / train_batches))
            print("  validation loss: %.10f" % val_loss[0])
            print("  validation acc:  %.10f" % val_acc )

            epoch = epoch + 1
        print("Max validation acc: %5.5f" % val_acc_max)
        print("")

        # print("W_val",W_val.shape)
        # remove bad neurons of last hidden layer
        # W_val, b_val = self.remove_bad_feature(self.W[-2].get_value(), self.b[-2].get_value(), threshold=self.pruning[-1])
        # self.W[-2].set_value(W_val)
        # self.b[-2].set_value(b_val)
        # # print("W_val",W_val.shape)
        # del W_val, b_val
        # rng = np.random
        # self.n_units_hidden[-1] = self.W[-2].get_value().shape[1]
        # self.W[-1].set_value(  np.asarray(rng.randn(self.n_units_hidden[-1], self.n_classes), dtype=self.mytype) )   # re-initialize W_out
        # # self.W[-1].set_value(  np.asarray(rng.randn(self.n_units_hidden[-1], self.n_classes), dtype=np.float64) )   # float64 avoid float32 underflow/overflow
        # # self.b[-1].set_value(  np.zeros(shape=(1,self.n_classes), dtype=np.float64)  )                              # float64 avoid float32 underflow/overflow
        # self.get_all_params_info()
        # # print(self.n_units_hidden)
        # # exit()
        #
        # import itertools
        # train_params = list(itertools.chain(*[self.W, self.b])) # Flatten a list of lists in one line
        # print("  train:", train_params)
        # print("  batch_size:%d n_epochs:%d learning_rate:%f" % (self.batch_size[-1], self.n_epochs[-1], self.learning_rate[-1]) )
        #     #  h = T.nnet.softmax(T.dot(self.a[self.n_hidden_layer-1], self.W[-1]) + self.b[-1] )
        # # self.h = T.nnet.softmax(T.dot(self.a[self.n_hidden_layer-1], self.W[-1]) + self.b[-1])        y_pred = T.argmax(h, axis=1)
        # ce = T.nnet.categorical_crossentropy(self.h, T.extra_ops.to_one_hot(self.targets, self.n_classes, dtype='int8')).mean()
        # # ce = T.clip(ce, 0, np.inf)  # when ce == nan, clip it to zero, avoid float32 underflow ? [still have NaN]
        # ce = ce + 1e-100     # avoid float32 underflow ? [still have NaN]
        # cost = ce
        # if self.update == 'adam':
        #     updates = self.adam(cost, train_params, learning_rate=self.learning_rate[-1])
        # elif self.update == 'fix':
        #     updates = self.fixupdate(cost, train_params, learning_rate=self.learning_rate[-1])
        # else:
        #     raise Exception("Unknow update method")
        # # X: Prof in the theano.function used to check GPU status via profmode.print_summary()
        # from theano import ProfileMode  # http://deeplearning.net/software/theano/tutorial/modes.html#using-profilemode
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # profmode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
        # # Nan
        # from theano.compile.nanguardmode import NanGuardMode
        # train_fn = theano.function(
        #           inputs=[self.inputs, self.targets],
        #           outputs=[cost],
        #           updates=updates,
        #         #   mode=profmode
        #         #  mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) # NaN debug
        #            )
        # test_fn = theano.function(
        #           inputs=[self.inputs, self.targets],
        #           outputs=[cost]
        #           )
        # acc = T.mean(T.eq(self.y_pred, self.targets), dtype=theano.config.floatX)
        # check_acc = theano.function(inputs=[self.inputs, self.targets], outputs=[acc, self.y_pred])
        #
        # import time
        # epoch = 0
        # val_acc_max = 0
        # for j in range(self.n_epochs[-1]):
        #     start_time = time.time()
        #     train_err = 0
        #     train_batches = 0
        #     for (X_train_a, y_train_a) in self.iterate_minibatches(X_train, y_train, self.batch_size[-1], shuffle=True):
        #         train_loss_val = train_fn(X_train_a, y_train_a)[0]
        #         # train_loss_val = np.nan
        #         # if train_loss_val == np.nan:    # debug NaN
        #             # print('  cost is NaN: float32 underflow?')
        #             # check_h_fn = theano.function(inputs=[self.inputs], outputs=self.h)
        #             # print('  h is:', check_h_fn(X_train_a))
        #             # print('  W_out:\n', self.W[-1].get_value())
        #             # print('    max(W_out):', np.max(self.W[-1].get_value()))
        #             # exit()
        #         train_err += train_loss_val
        #         train_batches += 1
        #         # print(train_loss_val.dtype) # float32
        #     # profmode.print_summary()
        #     # exit()
        #
        #     # _ , train_loss_val= train_fn(X_train, X_train)
        #     val_acc = check_acc(X_val, y_val)[0]
        #     if val_acc_max < val_acc:
        #         val_acc_max = val_acc
        #     # if epoch + 1 == 1 or (epoch + 1) % self.print_freq == 0:
        #     val_loss = test_fn(X_val, y_val)
        #     print("Epoch {} of {} took {:.3f}s".format(
        #                 epoch + 1, self.n_epochs[-1], time.time() - start_time))
        #     print("  training loss:   %.10f" % float(train_err / train_batches))
        #     print("  validation loss: %.10f" % val_loss[0])
        #     print("  validation acc:  %.10f" % val_acc )
        #
        #     epoch = epoch + 1
        # print("Max validation acc: %5.5f" % val_acc_max)
        # print("")

    def get_corrupted_input(self, input, corruption_level):
        pass

    def get_hidden_values(self, input):
        pass

    def get_reconstructed_input(self, hidden):
        pass

    def get_cost_updates(self, corruption_level, learning_rate):
        pass

    def get_all_params(self):
        params = []
        for key in self.layers:
            if key != 0:
                params.extend([self.layers[key].W, self.layers[key].b] )
        return params

    def get_all_params_value(self):
        params = self.get_all_params()
        params_value = []
        for i in range(len(params)):
            params_value.append(params[i].get_value())
        return params_value

    def get_all_params_info(self):
        params = self.get_all_params()
        params_value = self.get_all_params_value()
        for i in range(len(params_value)):
            print("   ", params[i], params_value[i].shape)
        for key in self.layers:
            if key != 0:
                print("   a", key, self.layers[key].a)

    def get_predict_function(self):
        self.predict_fn = theano.function(inputs=[self.inputs], outputs=self.y_pred)
        return self.predict_fn

    def adam(self, loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
        """Adam updates: see lasagne.updates.adam
        .. [1] Kingma, Diederik, and Jimmy Ba (2014):
               Adam: A Method for Stochastic Optimization.
               arXiv preprint arXiv:1412.6980.
        """
        all_grads = self.get_or_compute_grads(loss_or_grads, params)
        # t_prev = theano.shared(utils.floatX(0.))
        t_prev = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        from collections import OrderedDict
        updates = OrderedDict()

        t = t_prev + 1
        a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

        for param, g_t in zip(params, all_grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = beta1*m_prev + (1-beta1)*g_t
            v_t = beta2*v_prev + (1-beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates

    def fixupdate(self, loss_or_grads, params, learning_rate):
        """Stochastic Gradient Descent (SGD) updates
        Generates update expressions of the form:
        * ``param := param - learning_rate * gradient``
        Parameters
        ----------
        loss_or_grads : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        Returns
        -------
        OrderedDict
            A dictionary mapping each parameter to its update expression
        """
        grads = self.get_or_compute_grads(loss_or_grads, params)
        from collections import OrderedDict
        updates = OrderedDict()

        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad

        return updates

    @staticmethod   # no self
    def get_or_compute_grads(loss_or_grads, params):
        if any(not isinstance(p, theano.compile.SharedVariable) for p in params):
            raise ValueError("params must contain shared variables only. If it "
                             "contains arbitrary parameter expressions, then "
                             "lasagne.utils.collect_shared_vars() may help you.")
        if isinstance(loss_or_grads, list):
            if not len(loss_or_grads) == len(params):
                raise ValueError("Got %d gradient expressions for %d parameters" %
                                 (len(loss_or_grads), len(params)))
            return loss_or_grads
        else:
            return theano.grad(loss_or_grads, params)

    @staticmethod
    def remove_bad_feature(W, b, threshold=0.02):
        std_col = np.std(W,axis=0)
        remove_idx = np.where(std_col < threshold)  # 0.001
        # remove_idx = np.asarray([0,1,2,3])  # for debug
        # print('std_col:',std_col)
        print('    remove_idx:',remove_idx)
        W = np.delete(W, remove_idx, axis=1)
        b = np.delete(b, remove_idx, axis=0)

        if len(b) == 0:
            raise Exception('All neurons are removed. hint: set lower pi and/or pruning')
        return W, b

    @staticmethod
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle: # 打乱顺序
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @staticmethod
    def visualize_assquare_W(D1, n_units_l1, second=10, saveable=False, idx=None, fig_idx=0):
        plt.ion()
        fig = plt.figure(fig_idx)      # show all feature images
        size = D1.shape[0]

        num_r = int(np.round(np.sqrt(n_units_l1)))  # 每行显示的个数   若25个hidden unit -> 每行显示5个
        count = 1
        for row in range(1,num_r+1):
            for col in range(1, int(n_units_l1/num_r)+1):
                #print(col, row, count)
                a = fig.add_subplot(int(n_units_l1/num_r), num_r, count)
                # plt.imshow(np.reshape(D1.get_value()[:,count-1],(28,28)), cmap='gray')
                plt.imshow(np.reshape(D1[:,count-1] / np.sqrt( (D1[:,count-1]**2).sum()) ,(np.sqrt(size),np.sqrt(size))), cmap='gray', interpolation="nearest")
                plt.gca().xaxis.set_major_locator(plt.NullLocator())    # 不显示刻度(tick)
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                count = count + 1
        # plt.title('All feature groups from GVM')
        # plt.show()
        if saveable:
            plt.savefig(str(idx)+'.pdf',format='pdf')
        else:
            plt.draw()
            plt.pause(second)

def main():
    # rng = np.random.RandomState(123)
    # # rng = np.random.randn()
    # theano_rng = T.shared_randomstreams.RandomStreams(rng.randint(2 ** 30))
    # model = 'vanilaSSAE-2'
    model = 'SSBAE2'
    inputs = T.matrix('inputs', dtype='float32')    # DH: all theano SharedVariables : W1, W_out , must be float32, otherwise, GPU is disable even showing Using GPU on terminal
    targets = T.ivector('targets')
    if model == 'SSAE2':
        sae = StackedNet(
            # numpy_rng=rng,
            # theano_rng=theano_rng,
            inputs=inputs,
            targets=targets,
            n_visible=28 * 28,
            n_units_hidden=[400, 400],
            n_classes=10,
            W_exist=[],
            b_exist=[],
            n_epochs =[1000, 1000, 0],
            batch_size=[100, 100, 100],
            learning_rate=[0.0001, 0.0001, 0.0001],
            update='adam',
            dp=[0, 0, 0],
            beta=[4, 4],
            p_sparsity=[0.15*(400/400), 0.15*(400/400)],          # 0.3=0.15*(400/200)
            l2_lambda=[0.004*(400/400), 0.004*(400/400), 0.],
            l2_decoder_lambda=[0., 0.],
            pi=[0., 0.],
            pruning=[0, 0],
            uints_le_previous=False,
            print_freq=50,
        )
    elif model == 'SSBAE2':
        sae = StackedNet(   # 200->
            # numpy_rng=rng,
            # theano_rng=theano_rng,
            inputs=inputs,
            targets=targets,
            n_visible=28 * 28,
            n_units_hidden=[1600, 1600],
            n_classes=10,
            W_exist=[],
            b_exist=[],
            n_epochs =[1, 1, 1],
            batch_size=[100, 100, 100],
            learning_rate=[0.0001, 0.0001, 0.0001],
            update='adam',
            dp=[0, 0, 0],
            beta=[0, 4],
            p_sparsity=[0.15*(400/1600), 0.15*(400/1600)],              # 0.3=0.15*(400/200)
            l2_lambda=[0.0, 0.004*(400/1600), 0.],
            l2_decoder_lambda=[0.004*(400/1600), 0.0],     # 0.008=0.004*(400/200)
            pi=[4., 0.],
            pruning=[0.02, 0,],
            uints_le_previous=True,
            print_freq=50,
         )

    print("")



    X_train, y_train, _, _, X_test, y_test = load_dataset()
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)
    # X_train = X_train[::20]; y_train = y_train[::20]    # downsample
    # X_test = X_test[::20]; y_test = y_test[::20]    # downsample
    # print(X_train.shape)
    # print(X_test.shape)
    # exit()

    sae.start_greedy_layer_training(X_train, start=1)
    # sae.start_fine_turn_training(X_train, y_train, X_val, y_val)
    sae.start_fine_turn_training(X_train, y_train, X_test, y_test)

    print("\nEvaluation ...")
    predict_fn = sae.get_predict_function()
    y_predict = predict_fn(X_test)
    n_classes = 10
    c_mat, f1, acc, f1_macro = evaluation(y_test, y_predict, n_classes)

    print('\nSave whole model as < model_final_sae.mat > ...')
    import scipy
    params = sae.get_all_params()
    params_dict = {}
    for i in range(len(params)):
        params_dict[str(params[i])] = params[i].get_value()
    # print(params_dict)
    scipy.io.savemat('model_'+model+'.mat', params_dict)
    # scipy.io.savemat('model_sae.mat', {'W1':W1.get_value() , 'b1':b1.get_value(), 'W2':W2.get_value() , 'b2':b2.get_value(), 'W3':W3.get_value(), 'b3':b3.get_value()})

    print("\nAll properties of sae")
    attrs = vars(sae)
    print(', '.join("%s: %s\n" % item for item in attrs.items()))
    sae.get_all_params_info()




if __name__ == '__main__':
    print(theano.config.device)
    print(theano.config.force_device)
    theano.config.mode = 'FAST_RUN'     # The Function I Compiled is Too Slow, what’s up> http://deeplearning.net/software/theano/tutorial/debug_faq.html
    print(theano.config.mode)
    main()
