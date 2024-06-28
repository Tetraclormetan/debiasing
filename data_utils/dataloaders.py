import numpy as np
import jax
from torch.utils.data import DataLoader
import jax.numpy as jnp
from data_utils.CIFAR10C import CIFAR10C


@jax.jit
def collate_with_bias(batch):
  imgs = jax.numpy.stack([el[0] for el in batch])
  labels = jax.numpy.stack([el[1] for el in batch])
  biases = jax.numpy.stack([el[2] for el in batch])
  return imgs, labels, biases


def get_collate(shape):
    @jax.jit
    def collate_with_bias_and_resize(batch):
        imgs = jax.numpy.stack([el[0] for el in batch ])
        imgs = jax.image.resize(imgs, shape=(imgs.shape[0],) + shape, 
                                    method=jax.image.ResizeMethod.LINEAR)
        labels = jax.numpy.stack([el[1] for el in batch])
        biases = jax.numpy.stack([el[2] for el in batch])
        return imgs, labels, biases
    return collate_with_bias_and_resize


class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=True,
                timeout=0, worker_init_fn=None,
                collate_fn=collate_with_bias):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


class InMemoryDataset:
    def __init__(self, imgs, labels, biases,
                load_transform,
                rng_key,
                batch_size,
                weights):
        self.imgs = imgs
        self.labels = labels
        self.biases = biases
        self.load_transform = load_transform
        self.rng_key = rng_key
        self.batch_size = batch_size
        self.weights = weights
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return self.load_transform(self.imgs[i], subkey), self.labels[i], self.biases[i]
    
    def new_permutation(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        if self.weights is None:
            self.permutation = jax.random.permutation(subkey, len(self.labels))
        else:
            self.permutation = jax.random.choice(subkey, jnp.arange(len(self.labels)),
                                                (len(self.labels),), p=self.weights)
    
    def get_batch(self, i):
        subkeys = jax.random.split(self.rng_key, self.batch_size + 1)
        self.rng_key = subkeys[0]
        ids = self.permutation[i * self.batch_size: (1 + i) * self.batch_size]
        batch_transform = jax.vmap(self.load_transform, (0, 0), 0)
        return batch_transform(self.imgs[ids], subkeys[1:]), self.labels[ids], self.biases[ids]


def get_inmemory_dataset(dataset, rng_key, init_transform, load_transform, batch_size, weights):
        imgs = np.zeros((len(dataset),) + dataset[0][0].shape,dtype=float)
        labels = np.zeros((len(dataset)),dtype=int)
        biases = np.zeros((len(dataset)),dtype=int)
        for i, el in enumerate(dataset):
            imgs[i] = np.asarray(el[0])
            labels[i] = np.asarray(el[1])
            biases[i] = np.asarray(el[2])
        imgs = init_transform(imgs)
        return InMemoryDataset(imgs, labels, biases, load_transform, rng_key, batch_size, weights)
        

def get_dynamic_transform(input_shape, padding=4):
    def dynamic_transform(x, rng):
        out = x
        rnds = jax.random.randint(rng, shape=(4,), minval=jnp.array((0,0,0,0)), maxval=jnp.array((2*padding,2*padding,1,2))) 
        out = jax.lax.dynamic_slice(out, rnds[:3], input_shape[1:])
        out = jax.lax.cond(rnds[3] > 0, lambda x: x, lambda x: jax.numpy.flip(x, axis=1), out)
        return out
    return jax.jit(dynamic_transform)


def get_static_transform(padding=4):
    def static_full_dataset_transform(x):
        x = x / 255.
        x = jnp.pad(x, ((0,0),(padding,padding),(padding,padding),(0,0)))
        mean = jnp.array((0.4914, 0.4822, 0.4465))
        variance = jnp.square(jnp.array((0.2023, 0.1994, 0.2010)))
        x = jax.nn.standardize(x, mean=mean, variance=variance)
        return x
    return jax.jit(static_full_dataset_transform)


def get_cifar(dataset_config, stage_config, data_key):
    train_dataset = CIFAR10C(env="train", bias_amount=dataset_config['bias_amount'])
    train_dataset.transform = lambda x: jnp.asarray(x, dtype=np.float32)
    val_dataset = CIFAR10C(env="val", bias_amount=dataset_config['bias_amount'])
    val_dataset.transform = lambda x: jnp.asarray(x, dtype=np.float32)

    train_key, val_key = jax.random.split(data_key)

    train_data_weights = None
    if stage_config['use_weighted_sampling'] == True:
        correct_predictions = load_biases(stage_config['first_stage_result_path'])
        num_biased = correct_predictions.sum()
        num_unbiased = len(correct_predictions) - num_biased
        balancing_multiplier = num_biased / num_unbiased
        train_data_weights = jax.numpy.where(correct_predictions, 1, balancing_multiplier * stage_config["upsampling_constant"])

    train_inmemory = get_inmemory_dataset(dataset=train_dataset, rng_key=train_key, 
                        init_transform=get_static_transform(), 
                        load_transform=get_dynamic_transform(dataset_config['input_shape']),
                        batch_size=dataset_config["batch_size"],
                        weights=train_data_weights)
    val_inmemory = get_inmemory_dataset(dataset=val_dataset, rng_key=val_key, 
                        init_transform=get_static_transform(padding=0), 
                        load_transform=lambda x, _: x,
                        batch_size=dataset_config["batch_size"],
                        weights=None)
    
    val_loader = NumpyLoader(dataset=val_inmemory, batch_size=dataset_config['batch_size'], num_workers=0,
                            pin_memory=True)
    train_loader_sequential = NumpyLoader(dataset=train_inmemory, batch_size=dataset_config["batch_size"], num_workers=0,
                            pin_memory=True)
    
    return train_inmemory, val_loader, train_loader_sequential


def get_data_from_config(dataset_config, stage_config, data_key):
    if dataset_config["name"] == "CIFAR10C":
        data = get_cifar(dataset_config, stage_config, data_key)
    else:
        raise ValueError(f"Unknown dataset {dataset_config['name']}")
    return data


def save_biases(biases, path):
    with open(path, 'w') as file:
        pass
    with open(path, 'wb') as f:
        jnp.save(f, biases)


def load_biases(path):
    with open(path, "rb") as f:
        biases = jnp.load(f)
    return biases
