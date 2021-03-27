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

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01




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
       
    
    # create env, initialize the starting state
    env = gym.make("VizdoomHealthGathering-v0")
    #### INITIALIZE network with random weights
    
    # init replay buffer
    replay_buffer = ReplayBuffer(10000)

    for episode in tqdm(range(params["episodes"]), desc="episodes", unit="episodes"):

        done = False
        env.reset() #reset state

        # initialize frame stack
        frame_stack = deque(maxlen=params["stack_size"])

        # for frame skipping
        num_skipped = 0
        timestep = 0

        while not done: #for each time step
            env.render()

            # random action (for now)
            action = env.action_space.sample()

            #execute action and observe reward 
            # observation is screen info we want
            observation, reward, done, _ = env.step(action)

            ##STORE experience in replay memory###
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
            
            ####Sample Random Batch from replay memory####
            ###Preprocess states from this batch####
            ###Pass batch of preprocessed states to policy network###
            
            ### Calculate loss between q-actual and q-expected** #####
            ###Gradient descent update network weights####
            
            
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
