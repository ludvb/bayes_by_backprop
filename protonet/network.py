from functools import partial, reduce

import operator as op

from typing import List, Optional

import numpy as np

import torch as t
from torch.nn import init


class Linear(t.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weights = t.Tensor(in_features, out_features)
        self.bias = t.Tensor(out_features)

    def forward(self, x) -> t.Tensor:
        return (x @ self.weights) + self.bias


class MLELinear(Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.weights = t.nn.Parameter(self.weights)
        self.bias = t.nn.Parameter(self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        self.bias.data[...] = 0.


class BNNLinear(Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.w_mu = t.nn.Parameter(t.Tensor(*self.weights.shape))
        self.w_sd = t.nn.Parameter(t.Tensor(*self.weights.shape))

        self.b_mu = t.nn.Parameter(t.Tensor(*self.bias.shape))
        self.b_sd = t.nn.Parameter(t.Tensor(*self.bias.shape))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_mu, a=np.sqrt(5))
        self.b_mu.data[...] = 0.

        self.w_sd.data[...] = -100.
        self.b_sd.data[...] = -100.

    def forward(self, x: t.Tensor) -> t.Tensor:
        self.weights = (
            self.w_mu + t.log(1 + t.exp(self.w_sd)) * t.randn_like(self.w_sd))
        self.bias = (
            self.b_mu + t.log(1 + t.exp(self.b_sd)) * t.randn_like(self.b_sd))
        return super().forward(x)


def mle_loss(network, outputs, labels, class_weights):
    return -(
        t.sum(
            class_weights[labels]
            * t.log(outputs[range(len(outputs)), labels]),
        )
        /
        t.sum(class_weights[labels])
    )


def bnn_loss(network, outputs, labels, class_weights):
    return (
        mle_loss(network, outputs, labels, class_weights) +
        reduce(
            op.add,
            ((
                t.sum(
                    t.distributions.kl_divergence(
                        t.distributions.Normal(layer.w_mu, layer.w_sd),
                        t.distributions.Normal(0., 1.),
                    ))
                +
                t.sum(
                    t.distributions.kl_divergence(
                        t.distributions.Normal(layer.b_mu, layer.b_sd),
                        t.distributions.Normal(0., 1.),
                    ))
            ) for layer in network._forward if isinstance(layer, BNNLinear))
        ))


class MLP(t.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            num_hidden: int,
            hidden_size: int,
            bnn: bool = False,
            class_weights: Optional[List[float]] = None,
            activation: Optional[t.nn.Module] = None,
    ):
        super().__init__()

        self.class_weights = t.tensor(
            class_weights if class_weights
            else np.repeat(1., output_size)
        )

        if bnn:
            layer_type = BNNLinear
            self._loss = partial(
                bnn_loss,
                self,
                class_weights=self.class_weights,
            )
        else:
            layer_type = MLELinear
            self._loss = partial(
                mle_loss,
                self,
                class_weights=self.class_weights,
            )

        if activation is None:
            activation = t.nn.LeakyReLU(inplace=True)

        self._forward = t.nn.Sequential(
            layer_type(input_size, hidden_size),
            activation,
            *reduce(op.add, [[
                layer_type(hidden_size, hidden_size),
                activation,
            ] for _ in range(num_hidden)]),
            layer_type(hidden_size, output_size),
            t.nn.Softmax(dim=1),
        )

    def to(self, *args, **kwargs):
        self.class_weights.data = self.class_weights.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, x):
        inputs = x['input']
        labels = x['label']

        y = self._forward(inputs)

        return dict(
            prediction=y,
            accuracy=t.mean((t.argmax(y, dim=1) == labels).float()).item(),
            weighted_accuracy=(
                t.sum(
                    self.class_weights[labels]
                    * (t.argmax(y, dim=1) == labels).float(),
                ).item()
                /
                t.sum(self.class_weights[labels]).item()
            ),
            loss=self._loss(y, labels),
        )
