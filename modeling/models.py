# Implementation of model architectures
from flax import linen as nn
import jax.numpy as jnp
from flaxmodels import ResNet18


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


def get_model_and_variables(model_config_dict, init_key):
    if model_config_dict["name"] == "ResNet18":
        model = ResNet18(model_config_dict)
        variables = model.init(init_key, jnp.zeros(model_config_dict["input_shape"]))
    else:
        raise ValueError(f"Model not implemented {model_config_dict['name']}")
    return model, variables
