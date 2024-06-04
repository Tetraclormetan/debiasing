import jax
import jax.numpy as jnp
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


@jax.jit
def train_step_CE(state, batch):
  """Train for a single step."""
  image, label, _ = batch
  def loss_fn(params):
    logits, new_batch_stats = state.apply_fn({"params": params, 'batch_stats': state.batch_stats},
                            image, train=True, mutable=['batch_stats'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label).mean()
    return loss, new_batch_stats
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, new_batch_stats = grad_fn(state.params)
  state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats['batch_stats'])
  return state


@jax.jit
def train_step_GCE(state, batch, q=0.7):
  """Train for a single step."""
  images, labels, _ = batch
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
    return loss, new_batch_stats
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, new_batch_stats = grad_fn(state.params)
  state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats['batch_stats'])
  return state


@jax.jit
def compute_metrics(*, state: TrainStateWithStats, batch):
    images, labels, biases = batch
    logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, images,
                            train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()
    
    metric_updates = state.unmasked_metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss)
    unmasked_metrics = state.unmasked_metrics.merge(metric_updates)
    state = state.replace(unmasked_metrics=unmasked_metrics)
    
    conflicting_update = state.conflicting_accuracy.from_model_output(
        logits=logits, labels=labels, mask=biases - 1)
    conflicting_accuracy = state.conflicting_accuracy.merge(conflicting_update)
    state = state.replace(conflicting_accuracy=conflicting_accuracy)

    return state

# TODO: add metrics for unbiased?


def train(
    train_dataset: InMemoryDataset,
    val_loader,
    train_step,
    train_state: TrainStateWithStats,
    num_epochs,
    use_wandb=False,
):
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
            state = compute_metrics(state=state, batch=batch)

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

        early_stopping.update(metrics_history['val_loss'])
        if early_stopping.should_stop:
            print("Early stopping")
            break

        if use_wandb:
            wandb.log({key: val[-1] for key, val in metrics_history.items()})
        else:
            print(f"train epoch: {epoch + 1}, "
                f"loss: {metrics_history['train_loss'][-1]}, "
                f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
            print(f"test epoch: {epoch + 1}, "
                f"loss: {metrics_history['val_loss'][-1]}, "
                f"accuracy: {metrics_history['val_accuracy'][-1] * 100}")
    return state



def train_model_wandb(train_dataset, val_loader, train_step,  state, config_dict,
                      project_name=None):
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
        config_dict["num_epochs"],
        use_wandb=use_wandb,
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


def get_state_from_config(config, init_key):
    #resnet18 = ResNet18(output='logits', pretrained=None, normalize=False, num_classes=config["n_targets"])
    #variables = resnet18.init(init_key, jnp.zeros(config["input_shape"]))

    model, variables = get_model_and_variables(config["dataset"]["model"], init_key)

    optimizer = get_optimizer(config["optimizer"])
    return TrainStateWithStats.create(
        apply_fn = model.apply,
        params = variables['params'],
        tx = optimizer,
        batch_stats = variables['batch_stats'],
        unmasked_metrics=Metrics.empty(),
        conflicting_accuracy = metrics.Accuracy.empty()
    )
