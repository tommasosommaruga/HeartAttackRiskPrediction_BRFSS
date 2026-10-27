
"""
A small multilayer perceptron written in pure NumPy.

Building blocks:
    Linear -> BatchNorm1d -> ReLU  (repeated for each hidden layer) -> Linear

"""

import numpy as np


class Param:
    """A trainable array together with its gradient.

    Args:
        data: initial value of the parameter.
    """

    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


class Linear:
    """Fully connected layer: out = x @ W + b, with W of shape (in_dim, out_dim)."""

    def __init__(self, in_dim, out_dim, rng, bias=True):
        # He initialization, suited to ReLU activations
        w = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.W = Param(w)
        self.b = Param(np.zeros(out_dim)) if bias else None
        self.x = None

    def forward(self, x, training=True):
        self.x = x
        out = x @ self.W.data
        if self.b is not None:
            out = out + self.b.data
        return out

    def backward(self, grad_out):
        self.W.grad += self.x.T @ grad_out
        if self.b is not None:
            self.b.grad += grad_out.sum(axis=0)
        return grad_out @ self.W.data.T

    def parameters(self):
        return [self.W] if self.b is None else [self.W, self.b]


class ReLU:
    """ReLU activation: max(0, x)."""

    def __init__(self):
        self.mask = None

    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

    def parameters(self):
        return []


class BatchNorm1d:
    """Batch normalization over the batch dimension of a (N, D) input."""

    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = Param(np.ones(dim))
        self.beta = Param(np.zeros(dim))
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.momentum = momentum
        self.eps = eps
        self.x_hat = None
        self.inv_std = None

    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mean
            self.running_var = (1 - m) * self.running_var + m * var
        else:
            mean, var = self.running_mean, self.running_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = (x - mean) * self.inv_std
        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad_out):
        n = grad_out.shape[0]
        self.gamma.grad += (grad_out * self.x_hat).sum(axis=0)
        self.beta.grad += grad_out.sum(axis=0)
        # Gradient through the batch statistics (standard closed form)
        g = grad_out * self.gamma.data
        return (self.inv_std / n) * (
            n * g - g.sum(axis=0) - self.x_hat * (g * self.x_hat).sum(axis=0)
        )

    def parameters(self):
        return [self.gamma, self.beta]


class MLP:
    """Multilayer perceptron for binary classification (outputs one logit).

    Args:
        input_dim: number of input features.
        hidden_dims: list with the width of each hidden layer, e.g. [128, 64].
        batch_norm: if True, insert BatchNorm1d between each Linear and ReLU.
        seed: seed for the weight initialization.
    """

    def __init__(self, input_dim, hidden_dims, batch_norm=True, seed=42):
        self.config = {"input_dim": input_dim, "hidden_dims": list(hidden_dims), "batch_norm": batch_norm}
        rng = np.random.default_rng(seed)
        self.layers = []
        prev = input_dim
        for h in hidden_dims:
            # The bias is redundant when BatchNorm follows
            self.layers.append(Linear(prev, h, rng, bias=not batch_norm))
            if batch_norm:
                self.layers.append(BatchNorm1d(h))
            self.layers.append(ReLU())
            prev = h
        self.layers.append(Linear(prev, 1, rng))

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x[:, 0]

    def backward(self, grad_logits):
        grad = grad_logits[:, None]
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def state_dict(self):
        """Copy of everything learned: parameters and BatchNorm running stats."""
        state = {"param_{}".format(i): p.data.copy() for i, p in enumerate(self.parameters())}
        bns = [layer for layer in self.layers if isinstance(layer, BatchNorm1d)]
        for i, bn in enumerate(bns):
            state["bn_{}_mean".format(i)] = bn.running_mean.copy()
            state["bn_{}_var".format(i)] = bn.running_var.copy()
        return state

    def load_state_dict(self, state):
        """Restore the values returned by state_dict (on a model with the same shape)."""
        for i, p in enumerate(self.parameters()):
            p.data = state["param_{}".format(i)].copy()
        bns = [layer for layer in self.layers if isinstance(layer, BatchNorm1d)]
        for i, bn in enumerate(bns):
            bn.running_mean = state["bn_{}_mean".format(i)].copy()
            bn.running_var = state["bn_{}_var".format(i)].copy()

    def save(self, path):
        """Save weights and architecture to a .npz file."""
        np.savez(
            path,
            input_dim=self.config["input_dim"],
            hidden_dims=np.array(self.config["hidden_dims"]),
            batch_norm=self.config["batch_norm"],
            **self.state_dict()
        )

    @classmethod
    def load(cls, path):
        """Rebuild a model saved with save()."""
        d = np.load(path)
        model = cls(int(d["input_dim"]), [int(h) for h in d["hidden_dims"]], bool(d["batch_norm"]))
        model.load_state_dict(d)
        return model

    def predict_logits(self, x, batch_size=4096):
        """Raw outputs (logits) for each row of x, in evaluation mode."""
        logits = [
            self.forward(x[i : i + batch_size], training=False)
            for i in range(0, x.shape[0], batch_size)
        ]
        return np.concatenate(logits)

    def predict_proba(self, x):
        """Probability of the positive class for each row of x."""
        return 1.0 / (1.0 + np.exp(-self.predict_logits(x)))

    def predict_labels(self, x, threshold=0.5):
        """Predictions in {-1, 1}, the format required by create_csv_submission."""
        return np.where(self.predict_proba(x) >= threshold, 1, -1)
