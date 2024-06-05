import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import struct
from flax.training import train_state
from flax.core import FrozenDict
from flax.training.early_stopping import EarlyStopping
import optax
from clu import metrics
from tqdm import tqdm
import wandb 
from omegaconf import DictConfig

from modeling.models import get_model_and_variables
from modeling.optimizers import get_optimizer
from data_utils.dataloaders import InMemoryDataset


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss') # type: ignore


class TrainStateWithStats(train_state.TrainState):
        batch_stats: FrozenDict
        unmasked_metrics: Metrics
        conflicting_accuracy: metrics.Average

def metrics_from_logits(state, loss, logits, labels, biases):
    metric_updates = state.unmasked_metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss)
    unmasked_metrics = state.unmasked_metrics.merge(metric_updates)
    state = state.replace(unmasked_metrics=unmasked_metrics)
    
    conflicting_update = state.conflicting_accuracy.from_model_output(
        logits=logits, labels=labels, mask=biases - 1)
    conflicting_accuracy = state.conflicting_accuracy.merge(conflicting_update)
    state = state.replace(conflicting_accuracy=conflicting_accuracy)
    return state


@jax.jit
def train_step_CE(state, batch):
    """Train for a single step."""
    images, labels, biases = batch
    def loss_fn(params):
        logits, new_batch_stats = state.apply_fn({"params": params, 'batch_stats': state.batch_stats},
                                images, train=True, mutable=['batch_stats'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels).mean()
        return loss, (new_batch_stats, logits)
    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = value_and_grad_fn(state.params)
    new_batch_stats, logits = aux[1]
    loss = aux[0]
    state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats['batch_stats'])
    state = metrics_from_logits(state, loss, logits, labels, biases)
    return state


@jax.jit
def train_step_GCE(state, batch, q=0.7):
    """Train for a single step."""
    images, labels, biases = batch
    def loss_fn(params):
        logits, new_batch_stats = state.apply_fn({"params": params, 'batch_stats': state.batch_stats}, images,
                            train=True, mutable=['batch_stats'])
        logits_max = jnp.max(logits, axis=-1, keepdims=True)
        logits -= jax.lax.stop_gradient(logits_max)
        label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
        normalizers = jnp.sum(jnp.exp(logits), axis=-1)
    
        # outputs = jnp.exp(label_logits) / normalizers
        # loss = (1 - outputs**q).mean() / q
        log_probs_with_q = q * (label_logits - jnp.log(normalizers))
        loss = (1 - jnp.exp(log_probs_with_q).mean()) / q
        return loss, (new_batch_stats, logits)
    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = value_and_grad_fn(state.params)
    new_batch_stats, logits = aux[1]
    loss = aux[0]
    state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats['batch_stats'])
    state = metrics_from_logits(state, loss, logits, labels, biases)
    return state


@jax.jit
def compute_metrics(*, state: TrainStateWithStats, batch):
    images, labels, biases = batch
    logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, images,
                            train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()
    state = metrics_from_logits(state, loss, logits, labels, biases)
    return state


def train(
    train_dataset: InMemoryDataset,
    val_loader,
    train_step,
    train_state: TrainStateWithStats,
    num_epochs,
    checkpoint_path,
    use_wandb=False,
):
    use_checkpoints = checkpoint_path is not None
    if use_checkpoints:
        checkpoint_path = ocp.test_utils.erase_and_create_empty(checkpoint_path)
        options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
        checkpoint_manager = ocp.CheckpointManager(checkpoint_path, options=options)

    state = train_state

    metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'val_loss': [],
                   'val_accuracy': [],
                   'train_conflicting_accuracy': [],
                   'val_conflicting_accuracy': []
                   }
    
    early_stopping = EarlyStopping(min_delta=1e-4, patience=10)

    for epoch in tqdm(range(num_epochs), desc=f"{num_epochs} epochs", position=0, leave=True):
        train_dataset.new_permutation()
        for i in range(len(train_dataset) // train_dataset.batch_size):
            batch = train_dataset.get_batch(i)
            state = train_step(state, batch)

        for metric, value in state.unmasked_metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics
        metrics_history['train_conflicting_accuracy'].append(state.conflicting_accuracy.compute())

        state = state.replace(unmasked_metrics=state.unmasked_metrics.empty()) # reset train_metrics for next training epoch
        state = state.replace(conflicting_accuracy=state.conflicting_accuracy.empty())

        test_state = state
        for test_batch in val_loader:
            test_state = compute_metrics(state=test_state, batch=test_batch)

        for metric, value in test_state.unmasked_metrics.compute().items():
            metrics_history[f'val_{metric}'].append(value)
        metrics_history['val_conflicting_accuracy'].append(test_state.conflicting_accuracy.compute())

        test_state = test_state.replace(unmasked_metrics=state.unmasked_metrics.empty()) # reset train_metrics for next training epoch
        test_state = test_state.replace(conflicting_accuracy=state.conflicting_accuracy.empty())

        
        early_stopping.update(metrics_history['val_loss'][-1])
        if early_stopping.should_stop:
            print("Early stopping")
            break
        if use_checkpoints and \
            jnp.argmin(jnp.asarray(metrics_history['val_loss'])) == len(metrics_history['val_loss']) - 1:
                checkpoint_manager.save(epoch, args=ocp.args.StandardSave(train_state))

        if use_wandb:
            wandb.log({key: val[-1] for key, val in metrics_history.items()})
        else:
            print(f"train epoch: {epoch + 1}, "
                f"loss: {metrics_history['train_loss'][-1]}, "
                f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
            print(f"test epoch: {epoch + 1}, "
                f"loss: {metrics_history['val_loss'][-1]}, "
                f"accuracy: {metrics_history['val_accuracy'][-1] * 100}")
            
    if use_checkpoints:
        checkpoint_manager.wait_until_finished()
        restored_state_dict = checkpoint_manager.restore(checkpoint_manager.latest_step())
        state = state.replace(params=restored_state_dict['params'])
        state = state.replace(batch_stats=restored_state_dict['batch_stats'])
        state = state.replace(unmasked_metrics=Metrics.empty())
        state = state.replace(conflicting_accuracy=metrics.Accuracy.empty())
    return state



def train_model_wandb(train_dataset, val_loader, train_step, state, config_dict,
                      num_epochs, checkpoint_path, project_name=None):
    use_wandb = project_name is not None
    if use_wandb:
        wandb.init(
            project=project_name,
            config=flatten_tree(config_dict)
        )
    final_state = train(
        train_dataset,
        val_loader,
        train_step,
        state,
        num_epochs,
        use_wandb=use_wandb,
        checkpoint_path=checkpoint_path,
    )
    if use_wandb:
        wandb.finish()
    return final_state


def flatten_tree(tree_node, parent_key='', sep='_'):
    items = []
    for k, v in tree_node.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten_tree(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_state_from_config(dataset_config, optimizer_config, init_key):
    model, variables = get_model_and_variables(dataset_config["model"], init_key)

    optimizer = get_optimizer(optimizer_config)
    return TrainStateWithStats.create(
        apply_fn = model.apply,
        params = variables['params'],
        tx = optimizer,
        batch_stats = variables['batch_stats'],
        unmasked_metrics=Metrics.empty(),
        conflicting_accuracy = metrics.Accuracy.empty()
    )
