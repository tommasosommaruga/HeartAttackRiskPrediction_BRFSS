# -*- coding: utf-8 -*-
from helpers import *
"""
Gradient descent
"""
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y-tx@w
    return -(1/y.shape[0]) * (tx.T@e)
"""
MSE
"""
def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx@w
    loss = (1/(2*y.shape[0]))*(e@e)
    return loss

"""
Linear regression using gradient descent
"""
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape=(N,)
        tx: shape=(N, D)
        initial_w: shape=(D,)
        max_iters: number of iterations (int)
        gamma: learning rate (float)

    Returns:
        w: final weights, shape=(D,)
        loss: scalar MSE of the final weights
    """
    w = initial_w.copy()

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        loss = compute_loss(y, tx, w)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, compute_loss(y, tx, w)


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic
        gradient of the MSE loss at w.
    """
    return compute_gradient(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, )
        max_iters: number of SGD steps (int)
        gamma: learning rate (float)

    Returns:
        w: final weights, shape=(D,)
        loss: scalar MSE of the final weights
    """
    w = initial_w.copy()

    for n_iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_stoch_gradient(batch_y, batch_tx, w)
            w = w - gamma * grad
        loss = compute_loss(y, tx, w)
        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, compute_loss(y, tx, w)

def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)

    Returns:
        w: optimal weights, shape=(D,)
        loss: scalar MSE of the optimal weights
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using the closed-form solution.

    The regularization convention follows the course objective, where the
    squared-error term is averaged and the penalty is ``lambda_ * ||w||^2``.
    """
    a = tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def sigmoid(t):
    """apply sigmoid function on t. σ(x) := (1 + e−x)−1

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    # ***************************************************
    sigm = sigmoid(tx.dot(w))
    L = (-1/y.shape[0])*np.sum((y*np.log(sigm))+(1-y)*np.log(1-sigm))
    return L
    # ***************************************************

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    # ***************************************************
    return tx.T@(sigmoid(tx@w)-y)/y.shape[0]
    # ***************************************************

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    # ***************************************************
    grad = calculate_gradient(y,tx,w)
    loss = calculate_loss(y,tx,w)
    w =w-gamma*grad
    # ***************************************************
    return loss,w

def logistic_regression(y, tx, w, max_iters,gamma):
    """
    Logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N, 1). The labels (0 or 1).
        tx: numpy array of shape=(N, D). The feature matrix.
        w: numpy array of shape=(D, 1). The initial weight vector.
        max_iters: scalar. The maximum number of iterations.
        gamma: scalar. The learning rate (step size).

    Returns:
        w: numpy array of shape=(D, 1). The final weight vector.
        loss: scalar. The corresponding loss value.
    """
    threshold = 1e-8
    losses = []
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w,calculate_loss(y, tx, w)



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    # ***************************************************
    # return loss, gradient, and Hessian: 
    prod = tx.dot(w)
    # L = np.sum(-y@prod+np.log(1+np.exp(prod)))/y.shape[0]+(lambda_/2)*np.norm(w,ord=2)**2
    L = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)+2*lambda_*w
    
    # ***************************************************
    return L,grad



def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    # ***************************************************
    # return loss, gradient:
    loss , grad = penalized_logistic_regression(y,tx,w,lambda_)
    # update w:
    w = w-gamma*grad
    return loss, w


def reg_logistic_regression(y, tx,lambda_,w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent with L2 penalty.

    Args:
        y: numpy array of shape=(N, 1). The labels (0 or 1).
        tx: numpy array of shape=(N, D). The feature matrix.
        lambda_: scalar. The regularization parameter.
        w: numpy array of shape=(D, 1). The initial weight vector.
        max_iters: scalar. The maximum number of iterations.
        gamma: scalar. The learning rate (step size).

    Returns:
        w: numpy array of shape=(D, 1). The final weight vector.
        loss: scalar. The corresponding loss value (does not include the penalty term).
    """
    # init parameters
    threshold = 1e-8
    losses = []
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w,calculate_loss(y, tx, w)
    