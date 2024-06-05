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


def stage_pipeline(config_ : DictConfig, is_first_stage: bool) -> None:
    stage_name = "first_stage" if is_first_stage else "second_stage"

    dataset_config, stage_config, optimizer_config = \
        config_["dataset"], config_[stage_name], config_["optimizer"]
    config_for_logging = DictConfig({"dataset": dataset_config,
                                    stage_name: stage_config,
                                    "optimizer": optimizer_config })

    key_rng = jax.random.key(0)
    key_rng, model_key, data_key = jax.random.split(key_rng, 3)
    state = get_state_from_config(dataset_config, optimizer_config, model_key)
    train_dataset, val_loader, train_loader_sequential = \
        get_data_from_config(dataset_config, stage_config, data_key)
    
    if stage_config['loss']['name'] == 'GCE':
        train_step = jax.jit(partial(train_step_GCE, q=stage_config['loss']['GCE_q']))
    elif stage_config['loss']['name'] == 'CE':
        train_step = train_step_CE
    else:
        raise ValueError(f"Unexpected loss {stage_config['loss']['name']}")

    state = train_model_wandb(train_dataset, val_loader, train_step, state, config_for_logging,
                            num_epochs=stage_config['num_epochs'], 
                            checkpoint_path=stage_config['checkpoint_path'],
                            project_name=stage_config['name'])
    
    if is_first_stage and stage_config['save_predicted_errors']:
        biases = predict_bias(state, train_loader_sequential, len(train_dataset))
        save_biases(biases, stage_config['first_stage_result_path'])


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main_runner(config : DictConfig) -> None:
    print(config)
    if config['run_first_stage']:
        stage_pipeline(config, is_first_stage=True)
    if config['run_second_stage']:
        stage_pipeline(config, is_first_stage=False)


if __name__ == "__main__":
    main_runner()
