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