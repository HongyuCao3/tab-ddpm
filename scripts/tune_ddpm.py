import subprocess
import lib
import os
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true',  default=False)

args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type 
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)

pipeline = f'scripts/pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr
eval_seeds = f'scripts/eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)

def _suggest_mlp_layers(trial):
    """
    为给定的试验（trial）生成 MLP（多层感知机）的层配置建议。

    Args:
        trial: 一个试验对象，用于生成超参数建议。

    Returns:
        list: 包含各层维度（2的幂次方）的列表，表示 MLP 的层配置。

    Raises:
        ValueError: 如果生成的层数或维度不符合预期范围。
    """
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    """ 使用 Optuna 框架进行超参数优化的目标函数。

    Args: 
        trial (optuna.Trial): Optuna 提供的 Trial 对象，用于超参数采样和优化。

    Returns: 
        float: 多个数据集上的平均评分（R2 或 F1-score）。

    Raises: subprocess.CalledProcessError: 如果子进程执行失败。 FileNotFoundError: 如果配置文件或结果文件不存在。

    Notes: - 该函数通过修改基础配置并运行训练和评估流程来优化超参数。 - 每次试验会生成一个独立的实验目录，并在完成后清理。 - 支持多种评估指标（R2 或 F1-score），具体取决于数据集和模型类型。 """
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    weight_decay = 0.0    
    batch_size = trial.suggest_categorical('batch_size', [256, 4096])
    steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
    # steps = trial.suggest_categorical('steps', [500]) # for debug
    gaussian_loss_type = 'mse'
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

    base_config = lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')

    subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)

    n_datasets = 5
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        
        subprocess.run(['python3.9', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)

        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        report = lib.load_json(report_path)

        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            score += report['metrics']['val']['macro avg']['f1-score']

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

subprocess.run(['python3.9', f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample'], check=True)

if args.eval_seeds:
    best_exp = str(parent_path / f'{prefix}_best/config.toml')
    subprocess.run(['python3.9', f'{eval_seeds}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True)