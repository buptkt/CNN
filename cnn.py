import numpy as np
import time
from layers import *


class ThreeLayerConvNet:
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - bn - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
            self,
            input_dim=(3, 32, 32),
            num_filters=32,
            filter_size=7,
            hidden_dim=100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
            dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * input_dim[1] * input_dim[2] // 4, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['gamma'] = np.ones(hidden_dim)
        self.params['beta'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        gamma, beta = self.params['gamma'], self.params['beta']
        bn_params = {'mode': 'train'}
        if y is None:
            bn_params = {'mode': 'test'}

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        # 进行前向传播
        conv = Conv_fast()
        conv_naive = Conv_naive()
        relu1 = Relu()
        pool = MaxPool_fast()
        pool_naive = MaxPool_naive()
        affine1 = Affine()
        bn = BatchNorm()
        relu2 = Relu()
        affine2 = Affine()
        # start = time.time()
        conv.conv_forward_im2col(X, W1, b1, conv_param)
        relu1.relu_forward(conv.out)
        pool.max_pool_forward_fast(relu1.out, pool_param)
        # end = time.time()
        # print(f"time cost of fast_layers(conv - relu - pool)_forward is {end - start}")
        # start = time.time()
        # conv_naive.conv_forward_naive(X, W1, b1, conv_param)
        # relu1.relu_forward(conv_naive.out)
        # pool_naive.max_pool_forward_naive(relu1.out, pool_param)
        # end = time.time()
        # print(f"time cost of naive_layers(conv - relu - pool)_forward is {end - start}")
        affine1.affine_forward(pool.out, W2, b2)
        bn.batchnorm_forward(affine1.out, gamma, beta, bn_params)
        relu2.relu_forward(bn.out)
        affine2.affine_forward(relu2.out, W3, b3)
        scores = affine2.out

        if y is None:
            return scores

        # 进行反向传播
        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        affine2.affine_backward(dscores)
        relu2.relu_backward(affine2.dx)
        bn.batchnorm_backward(relu2.dx)
        affine1.affine_backward(bn.dx)
        # start = time.time()
        pool.max_pool_backward_fast(affine1.dx)
        relu1.relu_backward(pool.dx)
        conv.conv_backward_col2im(relu1.dx)
        # end = time.time()
        # print(f"time cost of fast_layers(conv - relu - pool)_backward is {end - start}")
        # start = time.time()
        # pool_naive.max_pool_backward_naive(affine1.dx)
        # relu1.relu_backward(pool_naive.dx)
        # conv_naive.conv_backward_naive(relu1.dx)
        # end = time.time()
        # print(f"time cost of naive_layers(conv - relu - pool)_backward is {end - start}")
        grads['W3'], grads['b3'] = affine2.dw, affine2.db
        grads['W2'], grads['b2'] = affine1.dw, affine1.db
        grads['W1'], grads['b1'] = conv.dw, conv.db
        # 别忘记正则项
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        grads['gamma'] = bn.dgamma
        grads['beta'] = bn.dbeta
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])
        loss += reg_loss

        return loss, grads
