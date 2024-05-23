# Starts training routine
import numpy as np
import jax

from data_utils.CMNIST import CMNIST
from data_utils.dataloaders import NumpyLoader
from modeling.config import CONFIG
from modeling.train_utils import train, train_step_CE
from modeling.models import create_simple_dnn_class


from flax.core import FrozenDict, frozen_dict
from flax.training import train_state
import jax.numpy as jnp
import optax
import warnings
from tqdm import tqdm
from torchvision import transforms

from modeling.train_utils import Metrics, TrainStateWithStats, compute_metrics
from modeling.models import get_model_and_variables
from modeling.optimizers import get_masked_optimizer


if __name__ == "__main__":

    key_rng = jax.random.key(0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model, variables = get_model_and_variables(18, CONFIG["input_shape"], CONFIG['n_targets'], 0)

    print(model.tabulate(jax.random.key(0), jnp.ones(CONFIG['input_shape']),
        compute_flops=False, compute_vjp_flops=False, depth=1, console_kwargs={'force_jupyter': False}))
    
    optimizer = get_masked_optimizer(CONFIG["opt_name"], CONFIG['opt_params'],
                                    variables, lambda s: s.startswith('backbone'))

    state = TrainStateWithStats.create(
        apply_fn = model.apply,
        params = variables['params'],
        tx = optimizer,
        batch_stats = variables['batch_stats'],
        metrics=Metrics.empty()
    )

    #parallel_train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,))
    #parallel_val_step = jax.pmap(val_step, axis_name='batch', donate_argnums=(0,))
    #parallel_test_step = jax.pmap(test_step, axis_name='batch', donate_argnums=(0,))
    #state = replicate(state)  # required for parallelism

    # control randomness on dropout and update inside train_step
    # key_rng, dropout_rng = jax.random.split(key_rng)  # for parallelism
    # print(jax.local_device_count())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=CONFIG['input_shape'][1:3], antialias=True),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CMNIST(env="train",bias_amount=0.95, 
                           transform=lambda x: np.asarray(transform(x).permute(1, 2, 0), dtype=np.float32))
    val_dataset = CMNIST(env="val",bias_amount=0.95, 
                         transform=lambda x: np.asarray(transform(x).permute(1, 2, 0), dtype=np.float32))

    train_loader = NumpyLoader(dataset=train_dataset, batch_size=CONFIG["batch_size"], num_workers=0)
    val_loader =  NumpyLoader(dataset=val_dataset,  batch_size=CONFIG["batch_size"], num_workers=0)


    for epoch_i in tqdm(range(CONFIG['num_epochs']), desc=f"{CONFIG['num_epochs']} epochs", position=0, leave=True):
        # training set
        train_loss, train_accuracy = [], []
        iter_n = len(train_loader)

        metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}
        
        with tqdm(total=iter_n, desc=f"{iter_n} iterations", leave=False) as progress_bar:
            for _batch in train_loader:
                batch, labels, biases =_batch
                # batch, labels = shard(batch), shard(labels)
                state = train_step_CE(state, _batch)
                #train_metadata = unreplicate(train_metadata)
                state = compute_metrics(state=state, batch=_batch)
                progress_bar.update(1)
        
        for metric, value in state.metrics.compute().items(): # compute metrics
            metrics_history[f'train_{metric}'].append(value) # record metrics
                
        print(f"train epoch: {epoch_i + 1}, "
              f"loss: {metrics_history['train_loss'][-1]}, "
              f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        
        # validation set
        
        # valid_accuracy = []
        # iter_n = len(test_dataset)
        # with tqdm(total=iter_n, desc=f"{iter_n} iterations", leave=False) as progress_bar:
        #     for _batch in test_dataset:
        #         batch = _batch[0]
        #         labels = _batch[1]

        #         batch = jnp.array(batch, dtype=jnp.float32)
        #         labels = jnp.array(labels, dtype=jnp.float32)

        #         batch, labels = shard(batch), shard(labels)
        #         metric = parallel_val_step(state, batch, labels)[0]
        #         valid_accuracy.append(metric)
        #         progress_bar.update(1)


        # avg_valid_acc = sum(valid_accuracy)/len(valid_accuracy)
        # avg_valid_acc = np.array(avg_valid_acc)[0]
        # print(f"[{epoch_i+1}/{Config['N_EPOCHS']}] Valid Accuracy: {avg_valid_acc:.03}")


