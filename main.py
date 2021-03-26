from collections import deque, namedtuple

import gym
import numpy as np
import torch
import vizdoomgym
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from replay_buffer import ReplayBuffer

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


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

    return params


def preprocess(observation):
    """
    Preprocess images to be used by DQN.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((60, 80))]
    )

    return transform(observation)


def update_stack(frame_stack, observation):
    image = preprocess(observation)

    frame_stack.append(image)


def optimize_dqn(replay_buffer, params):
    """
    Perform optimization on the DQN given a batch of randomly
    sampled transitions from the replay buffer.
    """

    if len(replay_buffer) < params["batch_size"]:
        return

    pass


def train(params):
    """
    Train the DQN. Assuming single episode, for now.
    """

    # create env
    env = gym.make("VizdoomHealthGathering-v0")

    # init replay buffer
    replay_buffer = ReplayBuffer(10000)

    for episode in tqdm(range(params["episodes"]), desc="episodes", unit="episodes"):

        done = False
        env.reset()

        # initialize frame stack
        frame_stack = deque(maxlen=params["stack_size"])

        # for frame skipping
        num_skipped = 0
        timestep = 0

        while not done:
            env.render()

            # random action (for now)
            action = env.action_space.sample()

            # observation is screen info we want
            observation, reward, done, _ = env.step(action)

            # only want to stack every four frames
            if (timestep == 0) or (num_skipped == params["skip_frames"] - 1):
                # reset counter
                num_skipped = 0
                stack_size = 0

                # get old stack, and update stack with current observation
                if len(frame_stack) > 0:
                    old_stack = torch.cat(tuple(frame_stack), axis=0)
                    stack_size, _, _ = old_stack.shape

                update_stack(frame_stack, observation)
                updated_stack = torch.cat(tuple(frame_stack), axis=0)

                # if old stack was full, we can store transition
                if stack_size == params["stack_size"]:
                    # store transition in replay buffer
                    replay_buffer.push(old_stack, action, updated_stack, reward)

            else:
                num_skipped += 1

            timestep += 1
    env.close()


def main():
    """
    Running the program.
    """
    # control random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    params = generate_params()

    for epoch in range(params["epochs"]):
        train(params)


if __name__ == "__main__":
    main()
