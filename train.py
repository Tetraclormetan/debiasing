# Starts training routine
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
import wandb

from data_utils.CMNIST import CMNIST
from data_utils.dataloaders import NumpyLoader
from modeling.config import CONFIG
from modeling.train_utils import train, train_step_CE, train_step_GCE, TrainStateWithStats, Metrics
from modeling.models import create_simple_dnn_class
from modeling.optimizers import get_optimizer


if __name__ == "__main__":
    key_rng = jax.random.key(0)
    key_rng, subkey = jax.random.split(key_rng)

    train_dataset = CMNIST(env="train", bias_amount=0.95, 
                           transform= lambda x: np.ravel(np.asarray(x, dtype=np.float32) / 255))
    val_dataset = CMNIST(env="val", bias_amount=0.95, 
                         transform= lambda x: np.ravel(np.asarray(x, dtype=np.float32) / 255))

    train_loader = NumpyLoader(dataset=train_dataset, batch_size=CONFIG["batch_size"], num_workers=0)
    val_loader =  NumpyLoader(dataset=val_dataset, batch_size=CONFIG["batch_size"], num_workers=0)

    DNN_CLASS = create_simple_dnn_class(CONFIG['n_targets'])
    dnn_model = DNN_CLASS()

    params = dnn_model.init(subkey, jnp.ones(CONFIG['input_shape']))["params"] # initialize parameters by passing a template image
    optimizer = get_optimizer(CONFIG['opt_name'], CONFIG['opt_params'])

    train_state = TrainStateWithStats.create(
        apply_fn=dnn_model.apply, 
        params=params,
        tx=optimizer,
        batch_stats=FrozenDict(),
        metrics=Metrics.empty())

    wandb.init(
        project="my-test-project",
        config=CONFIG
    )

    train(
        train_loader=train_loader,
        val_loader=val_loader,
        train_step=train_step_GCE,
        train_state=train_state,
        num_epochs=CONFIG["num_epochs"],
        use_wandb=True,
    )

    wandb.finish()
