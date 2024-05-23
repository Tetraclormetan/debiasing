# All the functionality for training
import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state
from flax.core import FrozenDict
import optax
from clu import metrics
from tqdm import tqdm, trange


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss') # type: ignore

# class TrainState(train_state.TrainState):
#     metrics: Metrics

class TrainStateWithStats(train_state.TrainState):
        batch_stats: FrozenDict
        metrics: Metrics


@jax.jit
def train_step_CE(state, batch):
  """Train for a single step."""
  image, label, _ = batch
  def loss_fn(params):
    logits = state.apply_fn({"params": params, 'batch_stats': state.batch_stats}, image)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=label).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state


@jax.jit
def train_step_GCE(state, batch, q=0.7):
  """Train for a single step."""
  images, labels, _ = batch
  def loss_fn(params):
    logits = state.apply_fn({"params": params, 'batch_stats': state.batch_stats}, images)
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)
    label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    normalizers = jnp.sum(jnp.exp(logits), axis=-1)
    
    # outputs = jnp.exp(label_logits) / normalizers
    # loss = (1 - outputs**q).mean() / q
    log_probs_with_q = q * (label_logits - jnp.log(normalizers))
    loss = (1 - jnp.exp(log_probs_with_q).mean()) / q
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

@jax.jit
def compute_metrics(*, state, batch):
    images, labels, _ = batch
    logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

# TODO: add metrics for unbiased?


def train(
    train_loader,
    val_loader,
    train_step,
    train_state,
    num_epochs,
):
    state = train_state #create_train_state(model, input_shape, init_rng, opt_name, opt_params)

    metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}

    for epoch in trange(num_epochs):
        for batch in tqdm(train_loader):
            state = train_step(state, batch)
            state = compute_metrics(state=state, batch=batch)

        for metric, value in state.metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics
        state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

        test_state = state
        for test_batch in tqdm(val_loader):
            test_state = compute_metrics(state=test_state, batch=test_batch)

        for metric,value in test_state.metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)

        print(f"train epoch: {epoch + 1}, "
              f"loss: {metrics_history['train_loss'][-1]}, "
              f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"test epoch: {epoch + 1}, "
              f"loss: {metrics_history['test_loss'][-1]}, "
              f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
