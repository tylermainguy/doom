##Imports
import numpy as np
import torch

from trainer import Trainer


def generate_params():
    """
    Generate parameters to be used by model.
    """
    params = {}

    params["timesteps"] = 1000
    params["episodes"] = 500
    params["epochs"] = 5
    params["stack_size"] = 4
    params["skip_frames"] = 4
    params["batch_size"] = 40
    params["device"] = "cpu"
    params["load_model"] = True

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

    for epoch in range(params["epochs"]):
        print("epoch {}".format(epoch))
        # trainer.train(epoch)
        trainer.evaluate()


if __name__ == "__main__":
    main()
