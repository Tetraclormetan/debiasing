# Implementation of model architectures
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
from jax import lax
from jax import random as jrandom
from flaxmodels import ResNet18
from typing import Callable, Optional, Any, Union, Tuple, Dict



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


class LinearBNN(nn.Module):
    in_features: int
    out_features: int
    param_dtype: Any = jnp.float32
    mu_init: Callable[[Any, Tuple[int, ...], Any], Any] = nn.initializers.ones #nn.lecun_normal()
    logvar_init: Callable[[Any, Tuple[int, ...], Any], Any] = nn.initializers.zeros
    
    def setup(self) -> None:
        self.kernel_mu = self.param('kernel_mu',
                            self.mu_init,
                            (self.in_features, self.out_features),
                            self.param_dtype)
        self.kernel_logvar = -3 * self.param('kernel_logvar',
                            self.logvar_init,
                            (self.in_features, self.out_features),
                            self.param_dtype)
        self.bias_mu = self.param('bias_mu', self.mu_init, (self.out_features,),
                            self.param_dtype)
        self.bias_logvar = -3 * self.param('bias_logvar', self.logvar_init, (self.out_features,),
                            self.param_dtype)
    
    def __call__(self, inputs: Any,
                 deterministic: Optional[bool] = None,
                 rng: Optional[Any] = None
                 ) -> Any:
        
        if not deterministic:
            kernel, bias = self.sample_params(rng)
        else:
            kernel, bias = self.kernel_mu, self.bias_mu
        
        return jnp.matmul(inputs, kernel) + bias
    
    def sample_params(self, rng: Optional[Any] = None):
        if rng is None:
            rng = self.make_rng('default')
        rng_kernel, rng_bias = jrandom.split(rng)
        kernel_eps = jrandom.normal(rng_kernel, shape=self.kernel_mu.shape)
        kernel = self.kernel_mu + kernel_eps * jnp.exp(self.kernel_logvar)
        bias_eps = jrandom.normal(rng_bias, shape=self.bias_mu.shape)
        bias = self.bias_mu + bias_eps * jnp.exp(self.bias_logvar)
        return kernel, bias


class ResnetLastBNN(nn.Module):
    resnet: nn.Module
    bnn: LinearBNN

    def __call__(self, inputs, train: bool):
        embeddings = self.resnet(inputs, train)
        output = self.bnn(embeddings, deterministic=False)
        return output
    
    def estimate_variation(self, inputs, train: bool, n_trials: int = 10):
        embeddings = self.resnet(inputs, train)
        results = jnp.zeros((n_trials, inputs.shape[0], self.bnn.out_features))
        for i in range(n_trials):
            results = results.at[i].set(self.bnn(embeddings))
        return results


def get_model_and_variables(model_config_dict, init_key):
    if model_config_dict["name"] == "ResNet18":
        model = ResNet18(**model_config_dict)
        variables = model.init(init_key, jnp.zeros(model_config_dict["input_shape"]))
    elif model_config_dict["name"] == "ResNet18BNN":
        resnet_key, bnn_key = jrandom.split(init_key)
        resnet = ResNet18(**model_config_dict["resnet"])
        print(model_config_dict,model_config_dict["resnet"])
        resnet_params = resnet.init(
            resnet_key, jnp.zeros(model_config_dict["input_shape"], jnp.float32))
        bnn = LinearBNN(
            in_features=model_config_dict["resnet"]["num_classes"],
            out_features=model_config_dict["num_classes"]
        )
        bnn_params = bnn.init(
            bnn_key, jnp.zeros((1, model_config_dict["resnet"]["num_classes"]), jnp.float32), rng=bnn_key)
        model = ResnetLastBNN(resnet, bnn)
        variables = FrozenDict({
            'params': {
                'resnet': resnet_params['params'],
                'bnn': bnn_params['params']
            },
            'batch_stats': {
                'resnet': resnet_params['batch_stats'],
                #'bnn': bnn_params['batch_stats']
            }
        })
    else:
        raise ValueError(f"Model not implemented {model_config_dict['name']}")
    return model, variables
