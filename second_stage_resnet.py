# Starts training routine
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import wandb
from torch.utils.data.sampler import WeightedRandomSampler
from torch import Generator


from data_utils.CIFAR10C import CIFAR10C
from data_utils.dataloaders import NumpyLoader
from modeling.config import CONFIG
from modeling.train_utils import train, train_step_CE, Metrics, TrainStateWithStats
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

    # choose weights to equalize number of biased and unbiased samples
    train_predicted_biases = jnp.load('data/bias.npy')
    part_biased = train_predicted_biases.mean()
    weights = train_predicted_biases * (1 - part_biased) + (1 - train_predicted_biases) * part_biased
    train_sampler =  WeightedRandomSampler(weights=np.asarray(weights),
                                           num_samples=len(train_dataset),
                                           generator=Generator().manual_seed(42),
                                           replacement=True)

    train_loader = NumpyLoader(dataset=train_dataset, sampler=train_sampler, 
                               batch_size=CONFIG["batch_size"], num_workers=0)
    val_loader =  NumpyLoader(dataset=val_dataset,  batch_size=CONFIG["batch_size"], num_workers=0)

    wandb.init(
        project="bias-second-stage",
        config=CONFIG
    )
    final_state = train(
        train_loader,
        val_loader,
        train_step_CE,
        state,
        CONFIG['num_epochs'],
        use_wandb=True,
    )
    wandb.finish()
