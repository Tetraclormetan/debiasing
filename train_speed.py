import jax
import jax.numpy as jnp
from torch.utils.data import RandomSampler
from torch import Generator
from tqdm import tqdm

from data_utils.CIFAR10C import CIFAR10C
from data_utils.dataloaders import get_inmemory_dataset, get_dynamic_transform, get_static_transform, NumpyLoader
from modeling.config import CONFIG
from modeling.train_utils import compute_metrics, train_step_CE, Metrics, TrainStateWithStats
from flaxmodels import ResNet18
from modeling.optimizers import get_optimizer


if __name__ == "__main__":
    key_rng = jax.random.key(0)

    train_dataset = CIFAR10C(env="train",bias_amount=0.95)
    train_dataset.transform = lambda x: jnp.asarray(x, dtype=jnp.float32)

    key_rng, train_key = jax.random.split(key_rng)
    key_rng, val_key = jax.random.split(key_rng)
    train_inmemory = get_inmemory_dataset(train_dataset, train_key, 
                        get_static_transform(), 
                        get_dynamic_transform(CONFIG['input_shape']))

    random_sampler = RandomSampler(train_inmemory, generator=Generator().manual_seed(42))
    train_loader = NumpyLoader(dataset=train_inmemory, batch_size=CONFIG["batch_size"], num_workers=0, 
                                pin_memory=True, sampler=random_sampler)
    
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

    batch = next(iter(train_loader))

    for _ in tqdm(range(len(train_loader))):
        state = train_step_CE(state, batch)
        state = compute_metrics(state=state, batch=batch)

