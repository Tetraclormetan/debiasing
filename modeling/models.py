# Implementation of model architectures
from flax import linen as nn
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
from jax_resnet import pretrained_resnet, slice_variables
from typing import Tuple


# DNN for CMNIST
def create_simple_dnn_class(output_dim: int):
    class SIMPLE_DNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(100)(x)
            x = nn.relu(x)
            x = nn.Dense(100)(x)
            x = nn.relu(x)
            x = nn.Dense(100)(x)
            x = nn.relu(x)
            x = nn.Dense(output_dim)(x)
            return x
    return SIMPLE_DNN


# ResNets
def get_head_class(output_dim: int):
    class Head(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = jnp.reshape(x, (x.shape[0], -1))
            x = nn.relu(x)
            x = nn.Dense(features=output_dim)(x)
            return x
    return Head

class StemHeadModel(nn.Module):
    '''Combines backbone and head model'''
    backbone: nn.Sequential
    head: nn.Module
        
    def __call__(self, inputs):
        x = self.backbone(inputs)
        x = self.head(x)
        return x

    
def _get_backbone_and_params(resnet_size: int = 18):
    ResNetClass, params = pretrained_resnet(resnet_size)
    model = ResNetClass()
    # get model & param structure for backbone
    start, end = 0, len(model.layers) - 2
    backbone = nn.Sequential(model.layers[start:end])
    backbone_params = slice_variables(params, start, end)
    return backbone, backbone_params


def get_model_and_variables(resnet_size: int, input_shape: Tuple[int,int,int,int], output_dim: int, head_init_key: int):

    #backbone
    inputs = jnp.ones(input_shape, jnp.float32)
    backbone, backbone_params = _get_backbone_and_params(resnet_size)
    key = jax.random.PRNGKey(head_init_key)
    backbone_output = backbone.apply(backbone_params, inputs, mutable=False)
    
    #head
    head_inputs = jnp.ones(backbone_output.shape, jnp.float32)
    head = get_head_class(output_dim=output_dim)()
    head_params = head.init(key, head_inputs)
    
    #final model
    model = StemHeadModel(backbone, head)
    variables = FrozenDict({
        'params': {
            'backbone': backbone_params['params'],
            'head': head_params['params']
        },
        'batch_stats': {
            'backbone': backbone_params['batch_stats'],
        }
    })
    return model, variables
