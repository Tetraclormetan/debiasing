# Starts training routine
import jax
import jax.numpy as jnp
import flax.linen as nn
import hydra
import optax
from omegaconf import DictConfig
from functools import partial

from modeling.train_utils import train_model_wandb, metrics_from_logits, get_state_from_config, compute_metrics, TrainStateWithStats
from data_utils.dataloaders import get_data_from_config, save_biases


def gaussian_kl(mu, logvar):
    kl_divergence = -jnp.sum(1 + 2 * logvar - mu**2 - jnp.exp(2 * logvar)) / 2
    return kl_divergence

def kl_single_bnn(params):
    kernel_kl = gaussian_kl(mu=params["kernel_mu"], logvar=params["kernel_logvar"])
    bias_kl = gaussian_kl(mu=params["bias_mu"], logvar=params["bias_logvar"])
    return kernel_kl + bias_kl

@jax.jit
def train_step_ELBO(state, batch, bnn_key, kl_discount=0.001):
    """Train for a single step."""
    bnn_train_key = jax.random.fold_in(key=bnn_key, data=state.step)
    images, labels, biases = batch
    def loss_fn(params):
        logits, new_batch_stats = state.apply_fn({"params": params, 'batch_stats': state.batch_stats},
                                images, train=True, mutable=['batch_stats'], rngs={'default':bnn_train_key})
        cross_entropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels).mean()
        kl_bnn = kl_single_bnn(params["bnn"])
        loss = cross_entropy - kl_discount * kl_bnn
        return loss, (new_batch_stats, logits)
    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = value_and_grad_fn(state.params)
    new_batch_stats, logits = aux[1]
    loss = aux[0]
    state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats['batch_stats'])
    state = metrics_from_logits(state, loss, logits, labels, biases)
    return state


def bnn_pipeline(config_ : DictConfig, is_first_stage: bool) -> None:
    stage_name = "first_stage"

    dataset_config, stage_config, optimizer_config = \
        config_["dataset"], config_[stage_name], config_["optimizer"]
    config_for_logging = DictConfig({"dataset": dataset_config,
                                    stage_name: stage_config,
                                    "optimizer": optimizer_config })

    key_rng = jax.random.key(0)
    key_rng, model_key, data_key = jax.random.split(key_rng, 3)
    state, model = get_state_from_config(dataset_config, optimizer_config, model_key)
    train_dataset, val_loader, train_loader_sequential = \
        get_data_from_config(dataset_config, stage_config, data_key)
    
    train_step = partial(train_step_ELBO, kl_discount=0.001)

    state = train_model_wandb(train_dataset, val_loader, train_step, state, config_for_logging,
                            num_epochs=stage_config['num_epochs'], 
                            checkpoint_path=stage_config['checkpoint_path'],
                            project_name=None)
    
    state = state.replace(apply_fn=partial(model.apply, method='estimate_variation'))
    entropys = []
    biases = []
    train_rng_key, key_rng = jax.random.split(key_rng)
    train_dataset.new_permutation()
    for i in range(len(train_dataset) // train_dataset.batch_size):
        batch = train_dataset.get_batch(i)
        train_key=jax.random.fold_in(train_rng_key, data=state.step)
        state, entropy, biase, labels = compute_distributions(state=state, batch=batch, rng=train_key)
        entropys.append(entropy)
        biases.append(biase)
    entropy = jnp.stack(entropys)
    bias = jnp.stack(biases)
    save_biases(bias, 'data/biases_.npy')
    save_biases(entropy, 'data/entropy_.npy')
    return state.conflicting_accuracy.compute()

@jax.jit
def compute_distributions(*, state: TrainStateWithStats, batch, rng):
    images, labels, biases = batch
    logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, images,
                            train=False, rngs={"default": rng})
    
    probs = nn.softmax(logits, axis = -1)
    entropy = jnp.mean(probs * jnp.log(probs), axis=[0, 2])
    return state, entropy, biases, labels

@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def bnn_runner(config : DictConfig) -> None:
    if config['run_first_stage']:
        bnn_pipeline(config, is_first_stage=True)


if __name__ == "__main__":
    bnn_runner()
