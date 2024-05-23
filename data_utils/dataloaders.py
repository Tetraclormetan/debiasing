import numpy as np
import jax
from torch.utils.data import DataLoader, default_collate

class NumpyCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=np.float32)) #[...,np.newaxis]

def numpy_collate(batch):
  return jax.tree_util.tree_map(np.asarray, default_collate(batch))

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)