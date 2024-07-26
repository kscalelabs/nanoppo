"""Proximal policy optimization training. Kindly lifted from the brax library.

See:
- https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
- https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from datetime import datetime
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from brax import base, envs
from brax.io import model
from brax.training import acting, gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
from omegaconf import DictConfig, OmegaConf
from orbax import checkpoint as ocp

from environment import HumanoidEnv

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


config = OmegaConf.create(
    {
        "num_timesteps": 150000000,
        "num_evals": 100,
        "reward_scaling": 0.1,
        "episode_length": 1000,
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": 20,
        "num_minibatches": 32,
        "num_updates_per_batch": 8,
        "discounting": 0.97,
        "learning_rate": 3e-4,
        "entropy_cost": 0.001,
        "num_envs": 4096,
        "batch_size": 512,
        "seed": 0,
        "policy_hidden_layer_sizes": [64, 64, 64, 64, 64],
        "value_hidden_layer_sizes": [256, 256, 256, 256, 256],
    }
)


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(
    v: Union[jnp.ndarray, running_statistics.RunningStatisticsState, Params],
) -> Union[jnp.ndarray, running_statistics.RunningStatisticsState, Params]:
    """Unpmaps a value.

    Args:
        v: The value to unmap.

    Returns:
        The value with the leading pmap axis removed.
    """
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(
    tree: Union[TrainingState, envs.State, types.Metrics],
) -> Union[TrainingState, envs.State, types.Metrics]:
    """Strips weak types from a tree structure.

    Args:
        tree: The tree structure (e.g., TrainingState, envs.State, or types.Metrics).

    Returns:
        The tree structure with weak types stripped.
    """

    # brax user code is sometimes ambiguous about weak_type.    in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf: Union[jnp.ndarray, Tuple[Any, ...]]) -> jnp.ndarray:
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = ppo_networks.make_ppo_networks,
    progress_fn: Callable[[int, types.Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    randomization_fn: Optional[Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]] = None,
    restore_checkpoint_path: Optional[str] = None,
    policy_hidden_layer_sizes: Sequence[int] = (256,) * 3,
    value_hidden_layer_sizes: Sequence[int] = (128,) * 3,
) -> Tuple[
    Callable[[Tuple[running_statistics.RunningStatisticsState, Params], bool], types.Policy],
    Tuple[running_statistics.RunningStatisticsState, Params],
    types.Metrics,
]:
    """PPO training.

    Args:
        environment: the environment to train
        num_timesteps: the total number of environment steps to use during training
        episode_length: the length of an environment episode
        action_repeat: the number of timesteps to repeat an action
        num_envs: the number of parallel environments to use for rollouts
            NOTE: `num_envs` must be divisible by the total number of chips since each
                chip gets `num_envs // total_number_of_chips` environments to roll out
            NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
                data generated by `num_envs` parallel envs gets used for gradient
                updates over `num_minibatches` of data, where each minibatch has a
                leading dimension of `batch_size`
        max_devices_per_host: maximum number of chips to use per host process
        num_eval_envs: the number of envs to use for evluation. Each env will run 1
            episode, and all envs run in parallel during eval.
        learning_rate: learning rate for ppo loss
        entropy_cost: entropy reward for ppo loss, higher values increase entropy of
            the policy
        discounting: discounting rate
        seed: random seed
        unroll_length: the number of timesteps to unroll in each environment. The
            PPO loss is computed over `unroll_length` timesteps
        batch_size: the batch size for each minibatch SGD step
        num_minibatches: the number of times to run the SGD step, each with a
            different minibatch with leading dimension of `batch_size`
        num_updates_per_batch: the number of times to run the gradient update over
            all minibatches before doing a new environment rollout
        num_evals: the number of evals to run during the entire training run.
            Increasing the number of evals increases total training time
        num_resets_per_eval: the number of environment resets to run between each
            eval. The environment resets occur on the host
        normalize_observations: whether to normalize observations
        reward_scaling: float scaling for reward
        clipping_epsilon: clipping epsilon for PPO loss
        gae_lambda: General advantage estimation lambda
        deterministic_eval: whether to run the eval with a deterministic policy
        network_factory: function that generates networks for policy and value
            functions
        progress_fn: a user-defined callback function for reporting/plotting metrics
        normalize_advantage: whether to normalize advantage estimate
        eval_env: an optional environment for eval only, defaults to `environment`
        policy_params_fn: a user-defined callback function that can be used for
            saving policy checkpoints
        randomization_fn: a user-defined callback function that generates randomized
            environments
        restore_checkpoint_path: the path used to restore previous model params
        policy_hidden_layer_sizes: unused
        value_hidden_layer_sizes: unused

    Returns:
        Tuple of (make_policy function, network params, metrics)
    """
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, " "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps / (num_evals_after_init * env_step_per_training_step * max(num_resets_per_eval, 1))
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    if isinstance(environment, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    # hopefully harmless
    T = TypeVar("T")

    def normalize(x: T, y: T) -> T:
        """Return x unchanged."""
        return x

    if normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        env_state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=(64,) * 5,
        value_hidden_layer_sizes=(64,) * 5,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    optimizer = optax.adam(learning_rate=learning_rate)

    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
    )

    gradient_update_fn = gradients.gradient_update_fn(loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

    def minibatch_step(
        carry: Tuple[optax.OptState, Params, PRNGKey],
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ) -> Tuple[Tuple[optax.OptState, Params, PRNGKey], types.Metrics]:
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params, normalizer_params, data, key_loss, optimizer_state=optimizer_state
        )

        return (optimizer_state, params, key), metrics

    def sgd_step(
        carry: Tuple[optax.OptState, Params, PRNGKey],
        unused_t: int,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ) -> Tuple[Tuple[optax.OptState, Params, PRNGKey], types.Metrics]:
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray) -> jnp.ndarray:
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t: int
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], types.Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy((training_state.normalizer_params, training_state.params.policy))

        def f(carry: Tuple[envs.State, PRNGKey], unused_t: int) -> Tuple[Tuple[envs.State, PRNGKey], types.Transition]:
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env, current_state, policy, current_key, unroll_length, extra_fields=("truncation",)
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f, (state, key_generate_unroll), (), length=batch_size * num_minibatches // num_envs
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params, data.observation, pmap_axis_name=_PMAP_AXIS_NAME
        )

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, types.Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step, (training_state, state, key), (), length=num_training_steps_per_epoch
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, types.Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch * env_step_per_training_step * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics  # pytype: disable=bad-return-type    # py311-upgrade

    # Initialize model params and training state.
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )

    training_state = TrainingState(  # jax-ndarray
        optimizer_state=optimizer.init(init_params),  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))),
        env_steps=0,
    )

    if num_timesteps == 0:
        return (
            make_policy,
            (training_state.normalizer_params, training_state.params),
            {},
        )

    if restore_checkpoint_path is not None and epath.Path(restore_checkpoint_path).exists():
        logging.info("restoring from checkpoint %s", restore_checkpoint_path)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        target = training_state.normalizer_params, init_params
        (normalizer_params, init_params) = orbax_checkpointer.restore(restore_checkpoint_path, item=target)
        training_state = training_state.replace(normalizer_params=normalizer_params, params=init_params)

    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])

    if not eval_env:
        eval_env = environment
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(randomization_fn, rng=jax.random.split(eval_key, num_eval_envs))
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )
    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, training_state.params.policy)), training_metrics={}
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = 0
    current_step = 0
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            breakpoint()
            current_step = int(_unpmap(training_state.env_steps))
            breakpoint()
            key_envs = jax.vmap(lambda x, s: jax.random.split(x[0], s), in_axes=(0, None))(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id == 0:
            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap((training_state.normalizer_params, training_state.params.policy)), training_metrics
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
            params = _unpmap((training_state.normalizer_params, training_state.params))
            policy_params_fn(current_step, make_policy, params)

    total_steps = current_step
    assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap((training_state.normalizer_params, training_state.params.policy))
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)


def main(config: DictConfig) -> None:
    env = HumanoidEnv()
    times = [datetime.now()]

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        times.append(datetime.now())

    def save_model(current_step: int, make_policy: str, params: dict[str, Any]) -> None:
        model_path = "model.pkl"
        model.save_params(model_path, params)
        print(f"Saved model at {current_step} to {model_path}")

    train(
        num_timesteps=config["num_timesteps"],
        num_evals=config["num_evals"],
        reward_scaling=config["reward_scaling"],
        episode_length=config["episode_length"],
        normalize_observations=config["normalize_observations"],
        action_repeat=config["action_repeat"],
        unroll_length=config["unroll_length"],
        num_minibatches=config["num_minibatches"],
        num_updates_per_batch=config["num_updates_per_batch"],
        discounting=config["discounting"],
        learning_rate=config["learning_rate"],
        entropy_cost=config["entropy_cost"],
        num_envs=config["num_envs"],
        batch_size=config["batch_size"],
        seed=config["seed"],
        policy_hidden_layer_sizes=config["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=config["value_hidden_layer_sizes"],
        environment=env,
        progress_fn=progress,
        policy_params_fn=save_model,
    )


if __name__ == "__main__":
    main(config=config)
