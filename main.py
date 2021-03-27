import math
import random
from collections import deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import vizdoomgym
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from dqn import DQN
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
    params["batch_size"] = 64

    return params


def preprocess(observation):
    """
    Preprocess images to be used by DQN.
    """
    # convert to grayscale, and reshape to be quarter of original size
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Grayscale(), transforms.Resize((60, 80))]
    )

    # return the transformed image
    return transform(observation)


def update_stack(frame_stack, observation):
    """
    Update the frame stack with the an observation
    """
    image = preprocess(observation)

    frame_stack.append(image)


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


def select_action(state, timesteps, policy_net, num_actions):

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * timesteps / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long)


def optimize_dqn(replay_buffer, target_net, pred_net, optim, params):
    """
    Perform optimization on the DQN given a batch of randomly
    sampled transitions from the replay buffer.

    Inputs:
        replay_buffer: buffer containing history of transitions
        params: dictionary containing parameters for network
    """

    # only want to update when we have enough transitions to sample
    if len(replay_buffer) < params["batch_size"] * 2:
        return

    # sample batch, and transpose batch
    sample = replay_buffer.sample(params["batch_size"])
    batch = Transition(*zip(*sample))

    # mask any transitions that have None (indicate transition to terminal state)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    # print(batch.action)
    # get batches of states, actions, and rewards for DQN
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # print("State size: {}".format(state_batch.shape))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print("Shape: {}".format(state_batch.shape))
    state_action_values = pred_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(params["batch_size"])
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA).unsqueeze(1) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    print("Loss: {}".format(loss.item()))
    # Optimize the model
    optim.zero_grad()
    loss.backward()
    for param in pred_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optim.step()


def train(params):
    """
    Train the DQN. Assuming single episode, for now.
    """

    env = gym.make("VizdoomHealthGathering-v0")
    num_actions = env.action_space.n

    target_net = DQN(60, 80, num_actions=num_actions)
    pred_net = DQN(60, 80, num_actions=num_actions)

    target_net.load_state_dict(pred_net.state_dict())  # create env, initialize the starting state
    target_net.eval()

    optim = torch.optim.Adam(pred_net.parameters())

    #### INITIALIZE network with random weights

    # init replay buffer
    replay_buffer = ReplayBuffer(10000)

    for episode in tqdm(range(params["episodes"]), desc="episodes", unit="episodes"):

        done = False
        env.reset()  # reset state

        # initialize frame stack
        frame_stack = deque(maxlen=params["stack_size"])

        # for frame skipping
        num_skipped = 0
        timestep = 0

        action = env.action_space.sample()

        while not done:
            env.render()

            # random action (for now)

            # execute action and observe reward
            # observation is screen info we want
            print("ACTION: {}".format(action))
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

                if not done:
                    updated_stack = torch.cat(tuple(frame_stack), axis=0)
                else:
                    # when we've reached a terminal state
                    updated_stack = None
                # if old stack was full, we can store transition
                if stack_size == params["stack_size"]:
                    # store transition in replay buffer
                    replay_buffer.push(
                        old_stack, torch.tensor([action]), updated_stack, torch.tensor([reward])
                    )

            else:
                num_skipped += 1

            if timestep % 100 == 0:
                optimize_dqn(replay_buffer, target_net, pred_net, optim, params)

            timestep += 1

            if len(frame_stack) == 4:
                action = select_action(
                    torch.cat(tuple(frame_stack)), timestep, pred_net, num_actions
                ).item()

            else:
                action = env.action_space.sample()
            if timestep % 1000 == 0:
                target_net.load_state_dict(pred_net.state_dict())
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
