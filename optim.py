import numpy as np

# 随机梯度下降
def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

# 动量更新，引入v，随着时间积累
def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    v = v * config['momentum'] - config['learning_rate'] * dw
    next_w = w + v

    config["velocity"] = v

    return next_w, config

# adagrad改进，引入decayrate，学习率不会单调减小，使用矩阵中每个数值的梯度平方和对梯度进行归一化
def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    grad_squ = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw * dw
    next_w = w - config['learning_rate'] * dw / (np.sqrt(grad_squ) + config['epsilon'])
    config['cache'] = grad_squ

    return next_w, config

# 可以看做是rmsprop和momentum_sgd的结合
def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    config['t'] += 1
    moment1 = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    moment2 = config['beta2'] * config['v'] + (1 - config['beta2']) * dw * dw
    # 两个bias来矫正momentum
    bias_1 = moment1 / (1 - config['beta1'] ** config['t'])
    bias_2 = moment2 / (1 - config['beta2'] ** config['t'])
    next_w = w - config['learning_rate'] * bias_1 / (np.sqrt(bias_2) + config['epsilon'])
    config['m'] = moment1
    config['v'] = moment2

    return next_w, config