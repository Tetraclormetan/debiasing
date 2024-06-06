import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import optuna

from runner import stage_pipeline


def real_objective(config: DictConfig) -> None:
    _ =   stage_pipeline(config, is_first_stage=True)
    res = stage_pipeline(config, is_first_stage=False)
    return res


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def optuna_runner(config : DictConfig) -> None:
    print(config)
    def get_optuna_objective():
        def run_optuna_objective(trial: optuna.Trial):
            new_config = config.copy()
            OmegaConf.set_struct(new_config, True)
            new_config.first_stage.num_epochs = trial.suggest_categorical("first_epochs", [5, 10, 20, 50, 100])
            new_config.second_stage.num_epochs = trial.suggest_categorical("second_epochs", [100])
            new_config.optimizer.init_params.learning_rate = trial.suggest_float("lr", 0.000001, 0.0001, log=True)
            new_config.dataset.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            new_config.first_stage.loss.GCE_q = trial.suggest_categorical("first_gce", [0.5, 0.7, 0.9])
            new_config.second_stage.upsampling_constant = trial.suggest_categorical("upsampling", [10, 20, 50, 100, 200])
            return real_objective(new_config)
        return run_optuna_objective
        
    objective = get_optuna_objective()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    exit()


if __name__ == "__main__":
    optuna_runner()
