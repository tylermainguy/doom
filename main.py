"""
Main module for training and evaluating our DQN agent in the doom environment.
"""

import argparse

import numpy as np
import torch

from trainer import Trainer


def generate_params() -> dict:
    """
    Generate parameters to be used by the model.

    Returns
    ----------
    dict
        Mapping for model parameters and values.

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=200000,
        help="Total number of episodes to train model on (default: 200,000).",
    )
    parser.add_argument(
        "--stack_size",
        type=int,
        default=4,
        help="Number of frames that will be stacked together to form a state (default: 4)",
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=4,
        help="Number of frames skipped before observation is used (default: 4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used to train the DQN (default: 64)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device the model is trained on (default: cpu)"
    )

    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discounted reward weighting (default: 0.99)"
    )
    parser.add_argument(
        "--eps_start",
        type=float,
        default=1.0,
        help="Initial epsilon value for training DQN with epsilon-greedy policy (default: 1.0)",
    )
    parser.add_argument(
        "--eps_end",
        type=float,
        default=0.1,
        help="Final epsilon value after linear annealing (default: 0.1)",
    )
    parser.add_argument(
        "--start_decay",
        type=int,
        default=100000,
        help="Learning step to begin epsilon decay (default: 100,000)",
    )
    parser.add_argument(
        "--end_decay",
        type=int,
        default=300000,
        help="Learning step to end epsilon decay (default: 300,000)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="defend",
        help="Game to train/evaluate model on (default: 'defend')",
    )
    parser.add_argument(
        "--eval", type=bool, default=False, help="Evaluate pretrained model (default: False)"
    )
    parser.add_argument(
        "--load_model",
        type=bool,
        default=False,
        help="Continue model training from checkpoint (default: False)",
    )

    return parser.parse_args()


def main():
    """
    Running the program.
    """
    # control random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    params = generate_params()

    # use GPUs when available
    if torch.cuda.is_available():
        params.device = "cuda:0"

    trainer = Trainer(params)

    if not params.eval:
        trainer.train()

    else:
        trainer.evaluate()


if __name__ == "__main__":
    main()
