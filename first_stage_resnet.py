# Starts training routine
import jax
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
from functools import partial

from modeling.train_utils import train_model_wandb, train_step_GCE, train_step_CE, get_state_from_config
from data_utils.dataloaders import get_data_from_config, save_biases


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


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def first_stage_pipeline(config : DictConfig) -> None:
    key_rng = jax.random.key(0)

    key_rng, resnet_key = jax.random.split(key_rng)
    state = get_state_from_config(config, resnet_key)

    key_rng, data_key = jax.random.split(key_rng)
    train_dataset, val_loader, train_loader_sequential = get_data_from_config(config, data_key)
    
    if config['stage']['name'] == "bias_identification":
        train_step = jax.jit(partial(train_step_GCE, q=config["GCE_q"]))
    else:
        train_step = train_step_CE

    state = train_model_wandb(train_dataset, val_loader, train_step, state, config, 
                              project_name=config['stage']['name'])
    
    if config['stage']['name'] == "bias_identification" and config['stage']['save_predicted_errors']:
        biases = predict_bias(state, train_loader_sequential, len(train_dataset))
        save_biases(biases, config['stage']['first_stage_result_path'])


if __name__ == "__main__":
    first_stage_pipeline()
