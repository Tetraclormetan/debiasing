import jax
import jax.numpy as jnp
from torch.utils.data import RandomSampler
from torch import Generator
from tqdm import tqdm

from data_utils.CIFAR10C import CIFAR10C
from data_utils.dataloaders import get_inmemory_dataset, get_dynamic_transform, get_static_transform, NumpyLoader
from modeling.config import CONFIG

if __name__ == "__main__":
    key_rng = jax.random.key(0)

    train_dataset = CIFAR10C(env="train",bias_amount=0.95)
    train_dataset.transform = lambda x: jnp.asarray(x, dtype=jnp.float32)
    val_dataset = CIFAR10C(env="val",bias_amount=0.95)
    val_dataset.transform = lambda x: jnp.asarray(x, dtype=jnp.float32)

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
    
    
    for batch in tqdm(train_loader):
        pass

    for batch in tqdm(val_loader):
        pass
