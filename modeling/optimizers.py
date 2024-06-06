import optax
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, frozen_dict
from typing import Dict

def get_optimizer(optimizer_dict: Dict):
    if optimizer_dict["name"] == "Adam":
        return optax.adam(**optimizer_dict["init_params"])
    elif optimizer_dict["name"] == "AdamW":
        return optax.adamw(**optimizer_dict["init_params"])
    elif optimizer_dict["name"] == "SGD":
        return optax.sgd(**optimizer_dict["init_params"])
    else:
        raise ValueError("optimizer not implemented")
        #return optax.sgd(**opt_params)

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

def get_masked_optimizer(opt_params: Dict, variables, mask_fn):
    optimizer = get_optimizer(opt_params)
    masked_optimizer = optax.multi_transform(
        {'non_zero': optimizer, 'zero': zero_grads()},
        create_mask(variables['params'], mask_fn)
    )
    return masked_optimizer