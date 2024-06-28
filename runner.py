# Starts training routine
import jax
import jax.numpy as jnp
from flax.linen import softmax
import hydra
from omegaconf import DictConfig
from functools import partial
from sklearn.metrics import classification_report

from modeling.train_utils import (train_model_wandb, get_train_step_from_config, 
                                  get_state_from_config, compute_metrics, TrainStateWithStats)
from data_utils.dataloaders import get_data_from_config, save_biases


@jax.jit
def predict_bias_step(state: TrainStateWithStats, batch, rng):
    images, labels, _ = batch
    logits = state.apply_fn({"params": state.params, 'batch_stats': state.batch_stats}, images, train=False)
    batch_predicted = (jnp.argmax(logits, -1) == labels).astype(int)
    return batch_predicted


@jax.jit
def predict_bias_bnn_step(state: TrainStateWithStats, batch, rng):
    images, labels, _ = batch
    logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, images,
                            train=False, rngs={"default": rng})
    probs = softmax(logits, axis = -1)
    # entropy = jnp.mean(probs * jnp.log(probs), axis=[0, 2])
    # label_probs = probs[jnp.arange(probs.shape[0])[:, None],
    #                     jnp.arange(probs.shape[1]), 
    #                     labels]
    # label_probs_mean = jnp.mean(label_probs, axis=0)
    # label_probs_std = jnp.std(label_probs, axis=0)
    batch_predicted = (jnp.argmax(jnp.mean(probs, axis=0), -1) == labels).astype(int)
    return batch_predicted


def predict_bias(state: TrainStateWithStats, model, train_loader_determ, train_len, is_bnn, rng_key):
    bias_predicted = jnp.zeros(train_len, dtype=int)
    bias = jnp.zeros(train_len, dtype=int)
    index = 0
    step_fn = predict_bias_bnn_step if is_bnn else predict_bias_step
    print(is_bnn)
    if is_bnn:
        state = state.replace(apply_fn=partial(model.apply, method='estimate_variation'))

    for batch in train_loader_determ:
        step_key = jax.random.fold_in(rng_key, data=state.step)
        _, _, batch_bias = batch
        batch_predicted = step_fn(state, batch, step_key)
        num_elements = len(batch[0])
        bias_predicted = bias_predicted.at[index: index + num_elements].set(batch_predicted)
        bias = bias.at[index: index + num_elements].set(batch_bias)
        index += num_elements

    print(classification_report)

    if is_bnn:
        state = state.replace(apply_fn=partial(model.apply))
    return bias_predicted


def stage_pipeline(config_ : DictConfig, is_first_stage: bool) -> None:
    stage_name = "first_stage" if is_first_stage else "second_stage"
    print(config_)
    dataset_config, stage_config, optimizer_config = \
        config_["dataset"], config_[stage_name], config_["optimizer"]
    config_for_logging = DictConfig({"dataset": dataset_config,
                                    stage_name: stage_config,
                                    "optimizer": optimizer_config })

    key_rng = jax.random.key(0)
    key_rng, model_key, data_key = jax.random.split(key_rng, 3)
    state, model = get_state_from_config(stage_config['model'], optimizer_config, model_key)
    train_dataset, val_loader, train_loader_sequential = \
        get_data_from_config(dataset_config, stage_config, data_key)
    
    train_step = get_train_step_from_config(stage_config['loss'])

    state = train_model_wandb(train_dataset, val_loader, train_step, state, config_for_logging,
                            num_epochs=stage_config['num_epochs'], 
                            checkpoint_path=stage_config['checkpoint_path'],
                            project_name=stage_config['name'])
    
    bias_predict_rng, key_rng = jax.random.split(key_rng)
    if is_first_stage and stage_config['save_predicted_errors']:
        biases = predict_bias(state, model, train_loader_sequential, len(train_dataset), stage_config["model"]['name']=='ResNet18BNN', bias_predict_rng)
        save_biases(biases, stage_config['first_stage_result_path'])
    
    for test_batch in val_loader:
        state = compute_metrics(state=state, batch=test_batch)
    
    return state.conflicting_accuracy.compute()


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main_runner(config : DictConfig) -> None:
    if config['run_first_stage']:
        stage_pipeline(config, is_first_stage=True)
    if config['run_second_stage']:
        stage_pipeline(config, is_first_stage=False)


if __name__ == "__main__":
    main_runner()
