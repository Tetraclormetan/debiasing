# Starts training routine
import numpy as np
import jax
import jax.numpy as jnp
import wandb

from data_utils.CIFAR10C import CIFAR10C
from data_utils.dataloaders import NumpyLoader, get_inmemory_dataset, get_static_transform, get_dynamic_transform
from modeling.config import CONFIG
from modeling.train_utils import train, train_step_GCE, Metrics, TrainStateWithStats
from modeling.optimizers import get_optimizer
from flaxmodels import ResNet18
from torch.utils.data import RandomSampler
from torch import Generator


if __name__ == "__main__":

    key_rng = jax.random.key(0)
    key_rng, resnet_key = jax.random.split(key_rng)

    resnet18 = ResNet18(output='ligits',
                   pretrained=None,
                   normalize=False,
                   num_classes=CONFIG["n_targets"])
    variables = resnet18.init(resnet_key, jnp.zeros(CONFIG["input_shape"]))

    optimizer = get_optimizer(CONFIG["opt_name"], CONFIG['opt_params'])

    state = TrainStateWithStats.create(
        apply_fn = resnet18.apply,
        params = variables['params'],
        tx = optimizer,
        batch_stats = variables['batch_stats'],
        metrics=Metrics.empty()
    )

    train_dataset = CIFAR10C(env="train",bias_amount=0.95)
    train_dataset.transform = lambda x: jnp.asarray(x, dtype=np.float32)
    val_dataset = CIFAR10C(env="val",bias_amount=0.95)
    val_dataset.transform = lambda x: jnp.asarray(x, dtype=np.float32)

    key_rng, train_key = jax.random.split(key_rng)
    key_rng, val_key = jax.random.split(key_rng)
    train_inmemory = get_inmemory_dataset(train_dataset, train_key, 
                        get_static_transform(), 
                        get_dynamic_transform(CONFIG['input_shape']))
    
    val_inmemory = get_inmemory_dataset(val_dataset, val_key, 
                        get_static_transform(padding=0), 
                        lambda x, _: x)

    random_sampler = RandomSampler(train_inmemory, generator=Generator().manual_seed(42))
    train_loader = NumpyLoader(dataset=train_inmemory, batch_size=CONFIG["batch_size"], num_workers=0, 
                               pin_memory=True, sampler=random_sampler)
    val_loader = NumpyLoader(dataset=val_inmemory, batch_size=CONFIG['batch_size'], num_workers=0, 
                             pin_memory=True)
    
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
        logits = state.apply_fn({"params": state.params, 'batch_stats': state.batch_stats}, images, train=False)
        batch_predicted = (jnp.argmax(logits, -1) == labels).astype(int)
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
