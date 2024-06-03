# Starts training routine
import jax
import jax.numpy as jnp
import wandb
import hydra
from omegaconf import DictConfig
from functools import partial

from modeling.train_utils import train, train_step_GCE, get_state_from_config, flatten_tree
from data_utils.dataloaders import get_data_from_config


def train_first_model(train_dataset, val_loader, state, config_dict, use_wandb):
    if use_wandb:
        wandb.init(
            project="bias-first-stage",
            config=flatten_tree(config_dict)
        )
    train_step = jax.jit(partial(train_step_GCE, q=config_dict["GCE_q"]))
    final_state = train(
        train_dataset,
        val_loader,
        train_step,
        state,
        config_dict["num_epochs"],
        use_wandb=use_wandb,
    )
    if use_wandb:
        wandb.finish()
    return final_state


@jax.jit
def predict_bias_step(state, batch):
    images, labels, _ = batch
    logits = state.apply_fn({"params": state.params, 'batch_stats': state.batch_stats}, images, train=False)
    batch_predicted = (jnp.argmax(logits, -1) == labels).astype(int)
    return batch_predicted


def predict_bias(state, train_loader_determ, train_len):
    bias_predicted = jnp.zeros(train_len, dtype=int)
    index = 0
    for batch in train_loader_determ:
        batch_predicted = predict_bias_step(state, batch)
        num_elements = len(batch[0])
        bias_predicted = bias_predicted.at[index: index + num_elements].set(batch_predicted)
        index += num_elements
    return bias_predicted


def save_biases(biases, path):
    with open(path, 'w') as file:
        pass
    with open(path, 'wb') as f:
        jnp.save(f, biases)


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def first_stage_pipeline(config : DictConfig) -> None:
    key_rng = jax.random.key(0)

    key_rng, resnet_key = jax.random.split(key_rng)
    state = get_state_from_config(config, resnet_key)

    key_rng, data_key = jax.random.split(key_rng)
    train_dataset, val_loader, train_loader_sequential = get_data_from_config(config, data_key)
    
    state = train_first_model(train_dataset, val_loader, state, config, use_wandb=True)
    biases = predict_bias(state, train_loader_sequential, len(train_dataset))

    save_biases(biases, config['biases_path'])


if __name__ == "__main__":
    first_stage_pipeline()
