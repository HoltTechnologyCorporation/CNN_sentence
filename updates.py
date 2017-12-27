import theano
import theano.tensor as T
import numpy as np

def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

def clip_norms(gs, c):
    norm = T.sqrt(sum(T.sum(T.sqr(g)) for g in gs))
    return [clip_norm(g, c, norm) for g in gs]


class Regularizer(object):

    def __init__(self, l1=0., l2=0., maxnorm=0.):
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            col_norms = p.norm(2, axis=0)
            desired = T.clip(col_norms, 0, maxnorm)
            p *= (desired / (1e-7 + col_norms))
        return p

    def gradient_regularize(self, p, g):
        g += p * self.l2
        g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        return self.max_norm(p, self.maxnorm)


class UpdateRule(object):

    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def updates(self, cost, params):
        raise NotImplementedError


class SGD(UpdateRule):
    """
    Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:
    * ``param := param - learning_rate * gradient``
    """

    def __init__(self, lr=0.01, *args, **kwargs):
        """
        Parameters
        ----------
        lr : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        UpdateRule.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def updates(self, cost, params):
        """
        Parameters
        ----------
        cost : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            updated_p = p - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class Momentum(UpdateRule):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        UpdateRule.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def updates(self, cost, params):
        """
        Parameters
        ----------
        cost : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m, v))

            updated_p = p + v
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class NAG(UpdateRule):

    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        UpdateRule.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def updates(self, cost, params):
        """
        Parameters
        ----------
        cost : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m,v))

            updated_p = p + self.momentum * v - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class RMSprop(UpdateRule):
    """
    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        UpdateRule.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def updates(self, cost, params):
        """
        Parameters
        ----------
        cost : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_t = self.rho * acc + (1 - self.rho) * T.sqr(g)
            updates.append((acc, acc_t))

            updated_p = p - self.lr * (g / T.sqrt(acc_t + self.epsilon))
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates

class Adam(UpdateRule):
    """
    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, *args, **kwargs):
        UpdateRule.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def updates(self, cost, params):
        """
        Parameters
        ----------
        cost : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        i = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        i_t = i + 1
        lr_t = self.lr * (T.sqrt(self.b2 ** i_t) / (self.b1 ** i_t))
        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)
            m_t = self.b1 * m + (1. - self.b1) * g
            v_t = self.b2 * v + (1. - self.b2) * T.sqr(g)
            g_t = m_t / (T.sqrt(v_t) + self.e)
            g_t = self.regularizer.gradient_regularize(p, g_t)
            p_t = p - lr_t * g_t
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((i, i_t))
        return updates

class Adagrad(UpdateRule):
    """
    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.
    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """

    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        UpdateRule.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def updates(self, cost, params):
        """
        Parameters
        ----------
        cost : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_t = acc + g ** 2
            updates.append((acc, acc_t))

            p_t = p - (self.lr / T.sqrt(acc_t + self.epsilon)) * g
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((p, p_t))
        return updates  

class Adadelta(UpdateRule):

    def __init__(self, rho=0.95, epsilon=1e-6, *args, **kwargs):
        """
        learning rate is 1.
        """
        UpdateRule.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def updates(self, cost, params):
        """
        Parameters
        ----------
        cost : symbolic expression or list of expressions
            A scalar loss expression, or a list of gradient expressions
        params : list of shared variables
            The variables to generate update expressions for
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        """
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)

            acc = theano.shared(p.get_value() * 0.)
            acc_delta = theano.shared(p.get_value() * 0.)
            acc_t = self.rho * acc + (1 - self.rho) * T.sqr(g)
            updates.append((acc,acc_t))

            update = g * T.sqrt(acc_delta + self.epsilon) / T.sqrt(acc_t + self.epsilon)
            updated_p = p - update
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))

            acc_delta_t = self.rho * acc_delta + (1 - self.rho) * T.sqr(update)
            updates.append((acc_delta,acc_delta_t))
        return updates
