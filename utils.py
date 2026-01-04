import math
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


def sparse_init(initializer, ratio):
    def init(
        key,
        shape,
        out_sharding,
    ):
        if len(shape) != 2:
            raise ValueError("Implemented only for 2D params.")

        dense_params = initializer(key, shape, out_sharding)

        fan_in, fan_out = shape
        num_zeros = int(math.ceil(ratio * fan_in))

        # We zero out num_zeros indices for each fan_out column, as in original implementation:
        # https://github.com/mohmdelsayed/streaming-drl/blob/407dca7a8b584c1c20bc649053557f66e270b1e6/sparse_init.py#L17
        zero_indices = jax.random.randint(
            key,
            shape=(num_zeros, fan_out),
            minval=0,
            maxval=fan_in,
        )
        sparse_params = dense_params.at[zero_indices, jnp.arange(fan_out)].set(0)

        return sparse_params

    return init


def l1_norm(tree):
    return jax.tree.reduce(
        operator.add,
        jax.tree.map(lambda t: jnp.abs(t).sum(), tree),
        initializer=0,
    )


class ObGDState(NamedTuple):
    traces: optax.Params


def obgd(
    learning_rate: jax.typing.ArrayLike = 1.0,
    gamma: jax.typing.ArrayLike = 0.99,
    lmbda: jax.typing.ArrayLike = 0.8,
    kappa: jax.typing.ArrayLike = 2.0,
) -> optax.GradientTransformationExtraArgs:
    def init_fn(params):
        traces = optax.tree.zeros_like(params)
        return ObGDState(traces=traces)

    def update_fn(updates, state, params=None, *, td_error, done, **extra_args):
        del extra_args
        new_traces = jax.tree.map(
            lambda t, g: gamma * lmbda * t + g, state.traces, updates
        )

        td_error_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        dot_product = (
            td_error_bar * l1_norm(new_traces) * learning_rate * kappa
        ).squeeze()

        step_size = learning_rate / jnp.maximum(dot_product, 1)
        updates = jax.tree.map(lambda t: step_size * td_error * t, new_traces)

        new_traces = jax.tree.map(lambda t: t * (1 - done), new_traces)
        return updates, ObGDState(traces=new_traces)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# NOTE: Apadtive variant implemented for convinenece, but not used here (and in original implementation)
class AdaptiveObGDState(NamedTuple):
    traces: optax.Params
    second_moments: optax.Params
    count: jax.typing.ArrayLike


def adaptive_obgd(
    learning_rate: jax.typing.ArrayLike = 1.0,
    gamma: jax.typing.ArrayLike = 0.99,
    lmbda: jax.typing.ArrayLike = 0.8,
    kappa: jax.typing.ArrayLike = 2.0,
    beta2: jax.typing.ArrayLike = 0.999,
    eps: jax.typing.ArrayLike = 1e-8,
) -> optax.GradientTransformationExtraArgs:
    def init_fn(params):
        traces = optax.tree.zeros_like(params)
        second_moments = optax.tree.zeros_like(params)
        return AdaptiveObGDState(traces=traces, second_moments=second_moments, count=0)

    def update_fn(updates, state, params=None, *, td_error, done, **extra_args):
        del extra_args
        # NOTE: better to use optax rmsprop utils like safe increment here...
        new_count = state.counter + 1
        new_traces = jax.tree.map(
            lambda t, g: gamma * lmbda * t + g, state.traces, updates
        )
        new_moments = jax.tree.map(
            lambda v, t: beta2 * v + (1.0 - beta2) * (td_error * t) ** 2,
            state.second_moments,
            new_traces,
        )
        new_moments_hat = jax.tree.map(
            lambda v: v / (1.0 - beta2**new_count), new_moments
        )
        new_traces_norm = jax.tree.map(
            lambda e, v: e / jnp.sqrt(v + eps), new_traces, new_moments_hat
        )

        td_error_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        dot_product = (
            td_error_bar * l1_norm(new_traces_norm) * learning_rate * kappa
        ).squeeze()

        step_size = learning_rate / jnp.maximum(dot_product, 1)
        updates = jax.tree.map(lambda t: step_size * td_error * t, new_traces_norm)

        new_traces = jax.tree.map(lambda t: t * (1 - done), new_traces)
        return updates, AdaptiveObGDState(
            traces=new_traces, second_moments=new_moments, count=new_count
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
