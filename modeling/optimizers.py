import optax
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, frozen_dict
from typing import Dict

def get_optimizer(name: str, opt_params: Dict):
    if name == "Adam":
        return optax.adam(**opt_params)
    elif name == "AdamW":
        return optax.adamw(**opt_params)
    else:
        return optax.sgd(**opt_params)

def zero_grads():
    '''
    Zero out the previous gradient computation
    '''
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_util.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

def create_mask(params, label_fn):
    def _map(params, mask, label_fn):
        for k in params:
            if label_fn(k):
                mask[k] = 'zero'
            else:
                if isinstance(params[k], FrozenDict):
                    mask[k] = {}
                    _map(params[k], mask[k], label_fn)
                else:
                    mask[k] = 'non_zero'
    mask = {}
    _map(params, mask, label_fn)
    return frozen_dict.freeze(mask)

def get_masked_optimizer(name: str, opt_params: Dict, variables, mask_fn):
    optimizer = get_optimizer(name, opt_params)
    masked_optimizer = optax.multi_transform(
        {'non_zero': optimizer, 'zero': zero_grads()},
        create_mask(variables['params'], mask_fn)
    )
    return masked_optimizer