# Global configuration for training

CONFIG = {
    "batch_size" : 32,
    "num_epochs" : 100,

    "input_shape" : (1, 32, 32, 3),
    "display_shape" : (28, 28),
    "n_targets" : 10,

    "GCE_q" : 0.7,
    "exp_avg" : 0.7,

    "opt_name" : "Adam",
    "opt_params" : {"learning_rate": 0.000005}, #"momentum" : 0.95
}