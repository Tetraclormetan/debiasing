# Starts training routine
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import wandb

from data_utils.CIFAR10C import CIFAR10C
from data_utils.dataloaders import NumpyLoader
from modeling.config import CONFIG
from modeling.train_utils import train, train_step_GCE, Metrics, TrainStateWithStats
from modeling.models import get_model_and_variables
from modeling.optimizers import get_masked_optimizer


if __name__ == "__main__":

    key_rng = jax.random.key(0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        #TODO: remove pretrained weights
        model, variables = get_model_and_variables(18, CONFIG["input_shape"], CONFIG['n_targets'], 0)

    optimizer = get_masked_optimizer(CONFIG["opt_name"], CONFIG['opt_params'],
                                    variables, lambda s: s.startswith('backbone'))

    state = TrainStateWithStats.create(
        apply_fn = model.apply,
        params = variables['params'],
        tx = optimizer,
        batch_stats = variables['batch_stats'],
        metrics=Metrics.empty()
    )

    @jax.jit
    def transform(x):
        out = x / 255
        out = jax.image.resize(out, shape=CONFIG['input_shape'][1:], 
                               method=jax.image.ResizeMethod.LINEAR)
        out = jax.nn.standardize(out, 
                                 mean=jnp.array((0.4914, 0.4822, 0.4465)),
                                 variance=jnp.square(jnp.array((0.2023, 0.1994, 0.2010))))
        return out

    train_dataset = CIFAR10C(env="train",bias_amount=0.95)
    train_dataset.transform = lambda x: transform(jnp.asarray(x, dtype=np.float32))
    val_dataset = CIFAR10C(env="val",bias_amount=0.95)
    val_dataset.transform = lambda x: transform(jnp.asarray(x, dtype=np.float32))

    train_loader = NumpyLoader(dataset=train_dataset, batch_size=CONFIG["batch_size"], num_workers=0)
    val_loader =  NumpyLoader(dataset=val_dataset, batch_size=CONFIG["batch_size"], num_workers=0)

    wandb.init(
        project="bias-first-stage",
        config=CONFIG
    )
    final_state = train(
        train_loader,
        val_loader,
        train_step_GCE,
        state,
        CONFIG['num_epochs'],
        use_wandb=True,
    )
    wandb.finish()

    @jax.jit
    def predict_bias_step(state, batch):
        images, labels, _ = batch
        logits = state.apply_fn({"params": state.params, 'batch_stats': state.batch_stats}, images)
        batch_predicted = (jnp.argmax(logits,-1) == labels).astype(int)
        return batch_predicted

    bias_predicted = jnp.zeros(len(train_dataset), dtype=int)
    index = 0
    for batch in train_loader:
        batch_predicted = predict_bias_step(final_state, batch)
        num_elements = len(batch[0])
        bias_predicted = bias_predicted.at[index: index + num_elements].set(batch_predicted)
        index += num_elements

    with open('data/bias.npy', 'w') as file:
        pass
    with open('data/bias.npy', 'wb') as f:
        jnp.save(f, bias_predicted)
