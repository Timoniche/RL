import os

import wandb
import yaml
from yaml import CLoader

from dqn import DQN
from trainer import train


def main():
    wandb_key = os.getenv('WANDB_KEY')
    wandb.login(key=wandb_key)

    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=CLoader)

    run = wandb.init(
        project=config['common']['project'],
        config=config,
        name=config['training_args']['run_name'],
    )

    dqn = DQN()
    train(dqn, run)


if __name__ == '__main__':
    main()
