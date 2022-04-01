import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    N = x.shape[0]
    # 注意这里改变x的维度时不要直接把x改变了，而是要用一个新的变量nx
    nx = x.reshape(N, -1)
    out = nx.dot(w) + b

    # cache中的应该是x，而不是改变维度后的nx
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    # 注意这里dx的维度要和forward中传进来的x的维度保持一致（这也是cache中要保存源x的原因，保持数据一致性）
    dx = dout.dot(w.T)
    dx = np.reshape(dx, x.shape)
    nx = x.reshape(x.shape[0], -1)
    dw = (nx.T).dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(x, 0)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = np.where(x > 0, dout, 0)

    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    # 计算loss
    N = x.shape[0]
    col_sum = np.sum(np.exp(x), axis=1)
    target = np.exp(x[range(N), y])
    loss = np.sum(-1 * np.log(target / col_sum)) / N

    # 计算梯度，参考交叉熵函数和链式求导法则
    dx = (np.exp(x).T / col_sum).T

    dx[range(N), y] -= 1
    dx = dx / N

    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, {}
    if mode == "train":
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        x_norm = (x - mean) / np.sqrt(var + eps)
        out = x_norm * gamma + beta
        # 这里要cache的变量很多，因为这些变量中都有x参与计算，因此在计算梯度时要使用到
        cache = (x, x_norm, gamma, beta, mean, var, eps)

    elif mode == "test":
        out = gamma * ((x - running_mean) / np.sqrt(running_var + eps)) + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    N = dout.shape[0]
    x, x_norm, gamma, beta, mean, var, eps = cache

    # 计算gamma和beta的梯度是因为这两者也是由x计算得来的，要用到求导的链式法则
    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx_norm = dout * gamma
    dvar = dx_norm * (-1 / 2) * np.power(var + eps, -3 / 2)
    dmean = dx_norm * (-1) / np.sqrt(var + eps)
    dx = dx_norm * np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N

    return dx, dgamma, dbeta


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.


    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modify the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride = conv_param['stride']
    pad = conv_param['pad']
    hh = w.shape[2]
    ww = w.shape[3]
    H = x.shape[2]
    W = x.shape[3]
    N = x.shape[0]
    F = w.shape[0]
    out_h = int(1 + (H + pad * 2 - hh) / stride)
    out_w = int(1 + (W + pad * 2 - ww) / stride)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, F, out_h, out_w))
    for i in range(N):
        for j in range(F):
            for k in range(out_h):
                for l in range(out_w):
                    out[i, j, k, l] = np.sum(
                        x_pad[i, :, k * stride:k * stride + hh, l * stride:l * stride + ww] * w[j, :, :, :]) + b[j]
    cache = (x, w, b, conv_param)

    return out, cache



def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    hh = w.shape[2]
    ww = w.shape[3]
    H = x.shape[2]
    W = x.shape[3]
    N = x.shape[0]
    F = w.shape[0]
    out_h = int(1 + (H + pad * 2 - hh) / stride)
    out_w = int(1 + (W + pad * 2 - ww) / stride)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    # 按照forward中的计算公式，可以理解成每个out中的元素都对应着一个区域的x和一个filter，因此反向时也要一一对应
    for i in range(dout.shape[0]):
        for j in range(dout.shape[1]):
            db[j] += np.sum(dout[i, j])
            for k in range(dout.shape[2]):
                for l in range(dout.shape[3]):
                    dx_pad[i, :, k * stride:k * stride + hh, l * stride:l * stride + ww] += dout[i, j, k, l] * w[j, :,
                                                                                                               :, :]
                    dw[j, :, :, :] += dout[i, j, k, l] * x_pad[i, :, k * stride:k * stride + hh,
                                                         l * stride:l * stride + ww]
    dx[:, :, :, :] = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db



def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    H = x.shape[2]
    W = x.shape[3]
    stride = pool_param['stride']
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    out_h = int(1 + (H - pool_h) / stride)
    out_w = int(1 + (W - pool_w) / stride)
    out = np.zeros((x.shape[0], x.shape[1], out_h, out_w))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(out_h):
                for l in range(out_w):
                    out[i, j, k, l] = np.max(x[i, j, k * stride:k * stride + pool_h, l * stride:l * stride + pool_w])

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    H = x.shape[2]
    W = x.shape[3]
    stride = pool_param['stride']
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    out_h = int(1 + (H - pool_h) / stride)
    out_w = int(1 + (W - pool_w) / stride)
    dx = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(out_h):
                for l in range(out_w):
                    max_v = np.max(x[i, j, k * stride:k * stride + pool_h, l * stride:l * stride + pool_w])
                    dx[i, j, k * stride:k * stride + pool_h, l * stride:l * stride + pool_w] = \
                        np.where(x[i, j, k * stride:k * stride + pool_h, l * stride:l * stride + pool_w] == max_v, dout[i, j, k, l], 0)

    return dx