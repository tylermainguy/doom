##Imports
import numpy as np
import torch

from trainer import Trainer


def generate_params():
    """
    Generate parameters to be used by model.
    """
    params = {}

    params["episodes"] = 200000
    params["epochs"] = 5
    params["stack_size"] = 4
    params["skip_frames"] = 4
    params["batch_size"] = 40
    params["device"] = "cpu"
    params["load_model"] = False
    params["gamma"] = 0.99
    params["eps_start"] = 1.0
    params["eps_end"] = 0.1
    params["eps_decay"] = 200

    return params


def main():
    """
    Running the program.
    """
    # control random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    params = generate_params()

    if torch.cuda.is_available():
        params["device"] = "cuda"
    trainer = Trainer(params)

    trainer.train()
    # trainer.evaluate()


if __name__ == "__main__":
    main()
