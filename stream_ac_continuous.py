from dataclasses import dataclass
from typing import NamedTuple

import distrax
import draccus
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import shimmy
from flax import nnx
from flax.nnx.nn.linear import default_kernel_init
from tqdm import trange

from utils import obgd, sparse_init
from wrappers import AddTimeInfo, NormalizeObservation, ScaleReward


class Actor(nnx.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, sparsity, rngs):
        kernel_init = sparse_init(default_kernel_init, ratio=sparsity)

        self.network = nnx.Sequential(
            nnx.Linear(obs_dim, hidden_dim, rngs=rngs, kernel_init=kernel_init),
            nnx.LayerNorm(hidden_dim, use_bias=False, use_scale=False, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, kernel_init=kernel_init),
            nnx.LayerNorm(hidden_dim, use_bias=False, use_scale=False, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(hidden_dim, act_dim * 2, rngs=rngs, kernel_init=kernel_init),
        )

    def __call__(self, x) -> distrax.Normal:
        out = self.network(x)
        mu, pre_std = jnp.split(out, 2, axis=-1)
        std = jax.nn.softplus(pre_std)
        return distrax.Normal(mu, std)


class Critic(nnx.Module):
    def __init__(self, obs_dim, hidden_dim, sparsity, rngs):
        kernel_init = sparse_init(default_kernel_init, ratio=sparsity)

        self.network = nnx.Sequential(
            nnx.Linear(obs_dim, hidden_dim, rngs=rngs, kernel_init=kernel_init),
            nnx.LayerNorm(hidden_dim, use_bias=False, use_scale=False, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, kernel_init=kernel_init),
            nnx.LayerNorm(hidden_dim, use_bias=False, use_scale=False, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(hidden_dim, 1, rngs=rngs, kernel_init=kernel_init),
        )

    def __call__(self, x):
        return self.network(x)


class Transition(NamedTuple):
    obs: jax.typing.ArrayLike
    action: jax.typing.ArrayLike
    reward: jax.typing.ArrayLike
    next_obs: jax.typing.ArrayLike
    done: jax.typing.ArrayLike


@jax.jit(static_argnames=["gamma", "entropy_coeff"])
def train_step(
    graphdef,
    params,
    transition: Transition,
    key: jax.random.PRNGKey,
    gamma: float,
    entropy_coeff: float,
):
    transition = jax.device_put(transition)
    actor, actor_optim, critic, critic_optim = nnx.merge(graphdef, params)

    def critic_loss_fn(model: Critic, s):
        return model(s).squeeze()

    value_pred, critic_grad = nnx.value_and_grad(critic_loss_fn)(critic, transition.obs)
    next_value_pred = critic(transition.next_obs)

    td_target = transition.reward + gamma * (1 - transition.done) * next_value_pred
    td_error = td_target - value_pred

    def actor_loss_fn(model: Actor, s, a):
        dist = model(s)
        log_prob = dist.log_prob(a).sum()
        entropy = entropy_coeff * dist.entropy().sum() * jnp.sign(td_error)
        return (log_prob + entropy).squeeze()

    actor_grad = nnx.grad(actor_loss_fn)(actor, transition.obs, transition.action)

    actor_optim.update(actor, actor_grad, td_error=td_error, done=transition.done)
    critic_optim.update(critic, critic_grad, td_error=td_error, done=transition.done)

    new_params = nnx.state((actor, actor_optim, critic, critic_optim))

    return new_params, key


@jax.jit
def sample_action(graphdef, params, obs, key):
    actor, *_ = nnx.merge(graphdef, params)
    key, _key = jax.random.split(key)
    dist = actor(jax.device_put(obs))
    action = dist.sample(seed=_key)
    return action, key


# NOTE: it may be a good idea to decouple actor and critic configs in your projects!
@dataclass
class TrainConfig:
    env_name: str = "dm_control/dog-run-v0"
    total_steps: int = 1_000_000
    hidden_dim: int = 128
    learning_rate: float = 1.0
    gamma: float = 0.99
    lmbda: float = 0.8
    kappa: float = 2.0
    entropy_coeff: float = 0.01
    init_sparsity: float = 0.9
    seed: int = 42


@draccus.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = ScaleReward(env, gamma=config.gamma)
    env = NormalizeObservation(env)
    env = AddTimeInfo(env)

    key = jax.random.key(config.seed)
    key, actor_key, critic_key = jax.random.split(key, num=3)

    actor = Actor(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        hidden_dim=config.hidden_dim,
        sparsity=config.init_sparsity,
        rngs=nnx.Rngs(actor_key),
    )
    actor_optim = nnx.Optimizer(
        actor,
        obgd(
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            lmbda=config.lmbda,
            kappa=config.kappa,
        ),
        wrt=nnx.Param,
    )

    critic = Critic(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=config.hidden_dim,
        sparsity=config.init_sparsity,
        rngs=nnx.Rngs(critic_key),
    )
    critic_optim = nnx.Optimizer(
        critic,
        obgd(
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            lmbda=config.lmbda,
            kappa=config.kappa,
        ),
        wrt=nnx.Param,
    )

    print(
        f"Actor params: {sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(actor, nnx.Param)))}"
    )
    print(
        f"Critic params: {sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(critic, nnx.Param)))}"
    )

    # splitting params into state and grapdef
    graphdef, params = nnx.split((actor, actor_optim, critic, critic_optim))

    obs, info = env.reset(seed=42)
    done = False
    pbar = trange(1, config.total_steps + 1)
    for t in pbar:
        if done:
            pbar.set_description(f"Episodic Return: {info['episode']['r']:.2f}")
            terminated, truncated = False, False
            obs, info = env.reset()

        action, key = sample_action(graphdef, params, obs, key)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        params, key = train_step(
            graphdef=graphdef,
            params=params,
            transition=Transition(*(obs, action, reward, next_obs, done)),
            key=key,
            gamma=config.gamma,
            entropy_coeff=config.entropy_coeff,
        )
        obs = next_obs


if __name__ == "__main__":
    train()
